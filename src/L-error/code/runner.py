from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone

EXPERIMENTS = ("Robusta-TPC-H", "Robusta-JOB")
HEADER = ["query_id", "default_plan_ms"] + [f"hint_{i:02d}_ms" for i in range(2, 50)] + ["l_error_ms", "speedup"]
TIMEOUT_LABEL = "\u8d85\u65f6"
PROTOCOL = "default3-min-success-hint1-sequential-dynamic-best-v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def natural_key(value):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", str(value))]


def write_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(6):
        try:
            tmp.replace(path)
            return
        except PermissionError as exc:
            if os.name != "nt" or exc.winerror not in (5, 32, 33) or attempt == 5:
                raise
            time.sleep(0.05 * 2**attempt)


def append_json(path, value):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def number(value):
    return "NA" if value is None else f"{value:.2f}"


def dynamic_timeout_ms(best):
    if not math.isfinite(best) or best < 0:
        raise ValueError("Best latency must be finite and nonnegative")
    return math.ceil(best + max(1000, 0.1 * best))


def cell(records):
    successful = [r["latency_ms"] for r in records if r.get("success")]
    if successful:
        return number(min(successful))
    if any(r.get("status") == "timeout" for r in records):
        return TIMEOUT_LABEL
    if any(r.get("attempted") for r in records):
        return "error"
    return "not_executed"


def summarize(query_id, records):
    groups = {i: [r for r in records if r["hint_id"] == i] for i in range(1, 50)}
    valid = {i: min(r["latency_ms"] for r in rs if r.get("success"))
             for i, rs in groups.items() if any(r.get("success") for r in rs)}
    default = valid.get(1)
    best_id = min(valid, key=lambda i: (valid[i], i)) if valid else None
    best = valid.get(best_id)
    error = default - best if default is not None else None
    speedup = default / best if default is not None and best is not None and best > 0 else None
    row = [query_id] + [cell(groups[i]) for i in range(1, 50)] + [number(error), number(speedup)]
    summary = dict(query_id=query_id, default_latency_ms=default, best_latency_ms=best,
                   best_hint_id=best_id, l_error_ms=error, speedup=speedup,
                   configuration_slots=49, actual_measurements=len(records),
                   successful_configurations=len(valid), default_statistic="min_successful_of_3",
                   status="completed" if default is not None else "default_unavailable")
    return row, summary


def is_link(path):
    if path.is_symlink():
        return True
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return os.name == "nt" and bool(info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def ensure_results(root):
    results = root / "results"
    if is_link(results):
        raise ValueError("The results directory must not be a link")
    results.mkdir(parents=True, exist_ok=True)
    if not results.is_dir():
        raise ValueError("The results path must be a directory")
    return results


def check_tree(path):
    if is_link(path):
        raise ValueError("Linked result paths are not allowed")
    if path.is_dir():
        for child in path.iterdir():
            check_tree(child)


def clean_output(root, experiment):
    if experiment not in EXPERIMENTS:
        raise ValueError("Unknown experiment")
    results = ensure_results(root)
    target = results / (experiment + "_results")
    if target.resolve().parent != results.resolve():
        raise ValueError("Result path escaped the results directory")
    if target.exists():
        check_tree(target)
        if not target.is_dir():
            raise ValueError("The experiment result path must be a directory")
        for child in target.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    target.mkdir(exist_ok=True)
    return target


@contextmanager
def runner_lock(results):
    path = results / ".runner.lock"
    if path.exists() and is_link(path):
        raise ValueError("Runner lock must not be a link")
    with path.open("a+b") as f:
        f.seek(0, 2)
        if f.tell() == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another package runner is active; no results were cleared") from exc
        try:
            yield
        finally:
            f.seek(0)
            if os.name == "nt":
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f, fcntl.LOCK_UN)


def load_queries(root, experiment):
    from core.validation import validate_select
    folder = root / experiment
    files = sorted(folder.glob("*.sql"), key=lambda p: natural_key(p.name))
    if not files:
        raise ValueError(f"No SQL files in {experiment}")
    result = []
    for path in files:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", path.stem):
            raise ValueError("SQL identifiers must use letters, digits, hyphens or underscores")
        data = path.read_bytes()
        sql = data.decode("utf-8-sig")
        validate_select(sql)
        result.append(dict(query_id=path.stem, file=f"{experiment}/{path.name}", sql_sha256=digest(data), sql=sql))
    return result


def connection_options(experiment, config):
    required = ("host", "port", "user", "password", "database")
    missing = [key for key in required if getattr(config, key, None) is None
               or (isinstance(getattr(config, key), str) and not getattr(config, key).strip())]
    if missing:
        raise ValueError("Fill Robusta/configuration-lerror.json before running: " + ", ".join(missing))
    if config.db_type != "postgresql":
        raise ValueError("This L-error runner requires db_type='postgresql'")
    if type(config.port) is not int or not 1 <= config.port <= 65535:
        raise ValueError("DB_CONFIG.port must be an integer from 1 to 65535")
    if type(config.connect_timeout) is not int or config.connect_timeout < 1:
        raise ValueError("DB_CONFIG.connect_timeout must be a positive integer")
    options = dict(host=config.host, port=config.port, user=config.user,
                   password=config.password, dbname=config.database,
                   connect_timeout=config.connect_timeout)
    schema = "tpch" if "TPC-H" in experiment else "job"
    options["options"] = f"-csearch_path={schema},public"
    options["application_name"] = "l_error_portable"
    return options


def run_query(connection, query, timeout_ms, out, measure=None):
    from core import database
    from core.hints import generate_configs
    measure = measure or database.execute_measurement
    configs = generate_configs()
    qid = query["query_id"]
    records = []
    best = None
    plan_dir = out / "plans" / qid
    plan_dir.mkdir(parents=True)
    checkpoint = out / "checkpoint.json"

    def execute(hint, repetition, limit):
        nonlocal best
        order = len(records) + 1
        write_json(checkpoint, dict(status="running", query_id=qid, hint_id=hint.hint_id,
                                   repetition=repetition, timeout_ms=limit, started_at=now()))
        started, clock = now(), time.monotonic()
        rec = measure(connection, query["sql"], hint, repetition, order, limit)
        rec.update(query_id=qid, sql_sha256=query["sql_sha256"], timeout_ms=limit,
                   t_best_before_ms=best, started_at=started, finished_at=now(),
                   wall_seconds=time.monotonic()-clock)
        plan = rec.pop("explain", None)
        if plan:
            relative = f"plans/{qid}/H{hint.hint_id:02d}_R{repetition}.json.gz"
            with gzip.open(out / relative, "wt", encoding="utf-8") as f:
                json.dump(plan, f)
            rec["plan_file"] = relative
            rec["planning_time_ms"] = plan[0].get("Planning Time")
        if not rec.get("success"):
            rec["latency_ms"] = None
        elif hint.hint_id > 1:
            best = min(best, rec["latency_ms"])
        rec["t_best_after_ms"] = best
        append_json(out / "measurements.jsonl", rec)
        records.append(rec)
        write_json(checkpoint, dict(status="measurement_saved", query_id=qid, hint_id=hint.hint_id,
                                   repetition=repetition, completed_measurements=len(records), updated_at=now()))
        if connection.closed or connection.broken or rec.get("cleanup_error"):
            raise RuntimeError("Database session lost or rollback failed; saved records retained")

    for repetition in range(1, 4):
        execute(configs[0], repetition, timeout_ms)
    successful = [r["latency_ms"] for r in records if r.get("success")]
    if successful:
        best = min(successful)
        for hint in configs[1:]:
            execute(hint, 1, dynamic_timeout_ms(best))
    row, summary = summarize(qid, records)
    summary["sql_sha256"] = query["sql_sha256"]
    append_json(out / "query_summaries.jsonl", summary)
    return row


def run_experiment(root, experiment, queries, timeout_ms, options):
    from core import database
    from core.hints import generate_configs
    connection = database.connect_database(options)
    try:
        settings = database.read_default_settings(connection, 10000)
        out = clean_output(root, experiment)
        versions = {p: importlib.metadata.version(p) for p in ("pglast", "psycopg")}
        write_json(out / "run.json", dict(experiment=experiment, protocol=PROTOCOL, started_at=now(),
            default_timeout_ms=timeout_ms, default_repetitions=3, hint_repetitions=1,
            hint_order=list(range(2, 50)), hint_timeout_formula="ceil(T_best + max(1000, 0.1*T_best))",
            default_settings=settings, packages=versions, postgresql_version=connection.info.server_version,
            queries=[{k:v for k,v in q.items() if k != "sql"} for q in queries]))
        write_json(out / "hint_configurations.json", [{"hint_id": h.hint_id, "settings":h.settings} for h in generate_configs()])
        with (out / "results.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            f.flush()
            for index, query in enumerate(queries, 1):
                message = f"{experiment}: {index}/{len(queries)} {query['query_id']}"
                print(message, flush=True)
                with (out / "run.log").open("a", encoding="utf-8") as log:
                    log.write(f"{now()} {message}\n")
                try:
                    row = run_query(connection, query, timeout_ms, out)
                except BaseException as exc:
                    write_json(out / "checkpoint.json", dict(status="interrupted", query_id=query["query_id"],
                        error_type=type(exc).__name__, finished_at=now(), note="Partial measurements are in measurements.jsonl"))
                    raise
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())
        write_json(out / "checkpoint.json", dict(status="completed", queries=len(queries), finished_at=now()))
    finally:
        connection.close()


def check_connections(names, config):
    from core import database
    checked = {}
    for name in names:
        schema = "tpch" if "TPC-H" in name else "job"
        connection = database.connect_database(connection_options(name,config))
        try:
            with connection.cursor() as cursor:
                cursor.execute("BEGIN READ ONLY")
                cursor.execute("SET LOCAL statement_timeout = 10000")
                cursor.execute("SELECT current_setting('server_version'), current_schema(), "
                               "has_schema_privilege(%s, 'USAGE')", (schema,))
                version, actual_schema, permitted = cursor.fetchone()
                if actual_schema != schema or not permitted:
                    raise ValueError(f"Required schema is unavailable: {schema}")
                cursor.execute("SELECT count(*), count(*) FILTER (WHERE NOT has_table_privilege(c.oid, 'SELECT')) "
                               "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
                               "WHERE n.nspname=%s AND c.relkind IN ('r','p')", (schema,))
                tables, inaccessible = cursor.fetchone()
                if not tables or inaccessible:
                    raise ValueError(f"Required tables or SELECT privileges are unavailable: {schema}")
                checked[name] = dict(server_version=version, schema=schema, tables=tables,
                                     select_privileges=True)
        finally:
            try:
                connection.rollback()
            finally:
                connection.close()
    return checked


def l_error_report(source):
    with (source / "query_summaries.jsonl").open(encoding="utf-8") as f:
        summaries = [json.loads(line) for line in f if line.strip()]
    values = [s["l_error_ms"] for s in summaries if s["l_error_ms"] is not None]
    maximum = max(values) if values else None
    average = math.fsum(values) / len(values) if values else None
    lines = [f"Maximum L-error (ms): {number(maximum)}",
             f"Average L-error (ms): {number(average)}",
             "", "L-error per SQL (ms; NA excluded from maximum and average):"]
    lines.extend(f"{s['query_id']}.sql: {number(s['l_error_ms'])}" for s in summaries)
    return "\n".join(lines) + "\n"


def archive_results(root, names):
    root = Path(root).resolve()
    source = root / "results"
    data_dir = root.parents[1] / "data"
    archive = data_dir / "L-error_results"
    if data_dir.resolve() != data_dir or archive.resolve() != archive:
        raise ValueError("The L-error archive must remain inside the project data directory")
    if is_link(data_dir) or is_link(archive) or is_link(source):
        raise ValueError("Linked L-error archive paths are not allowed")
    folders = []
    for name in names:
        if name not in EXPERIMENTS:
            raise ValueError("Unknown experiment to archive")
        folder = name + "_results"
        current, target = source / folder, archive / folder
        check_tree(current)
        check_tree(target)
        if not current.is_dir() or (target.exists() and not target.is_dir()):
            raise ValueError("L-error result and archive paths must be directories")
        checkpoint = json.loads((current / "checkpoint.json").read_text(encoding="utf-8"))
        if checkpoint.get("status") != "completed":
            raise ValueError(f"Cannot archive an incomplete L-error experiment: {name}")
        folders.append(folder)
    legacy_events = archive / "launcher_events.jsonl"
    check_tree(legacy_events)
    if legacy_events.exists() and not legacy_events.is_file():
        raise ValueError("The legacy launcher event archive must be a file")

    archive.mkdir(parents=True, exist_ok=True)
    # Prepare both reports for every selected experiment before clearing old archives.
    with tempfile.TemporaryDirectory(prefix=".archive-", dir=archive) as temporary:
        staging = Path(temporary)
        for folder in folders:
            staged = staging / folder
            staged.mkdir()
            shutil.copy2(source / folder / "results.csv", staged / "results.csv")
            (staged / "l_error_summary.txt").write_text(l_error_report(source / folder), encoding="utf-8")
        for folder in folders:
            target = archive / folder
            if target.resolve().parent != archive or is_link(target):
                raise ValueError("Refusing to clear an archive outside L-error_results")
            check_tree(target)
            target.mkdir(exist_ok=True)
            for child in target.iterdir():
                if child.resolve().parent != target:
                    raise ValueError("Refusing to clear a redirected archive entry")
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            for child in (staging / folder).iterdir():
                child.replace(target / child.name)
        if legacy_events.exists():
            if legacy_events.resolve().parent != archive or is_link(legacy_events):
                raise ValueError("Refusing to remove a redirected launcher event archive")
            legacy_events.unlink()
    return archive


def main(root, selected, timeouts, config, argv=None):
    parser = argparse.ArgumentParser(description="Run serial PostgreSQL L-error experiments.")
    parser.add_argument("--experiment", choices=("all",)+EXPERIMENTS, default=selected)
    parser.add_argument("--validate-only", action="store_true", help="Check SQL and protocol without connecting or clearing results")
    parser.add_argument("--check-connection", action="store_true", help="Check SQL, connection and schema permissions without workload measurements or output changes")
    args = parser.parse_args(argv)
    names = EXPERIMENTS if args.experiment == "all" else (args.experiment,)
    results = root / "results"
    safe_events = False
    try:
        if args.experiment not in ("all",)+EXPERIMENTS:
            raise ValueError("Unknown experiment selection in configuration-lerror.json")
        data = {name:load_queries(root,name) for name in names}
        for name in names:
            if not isinstance(timeouts[name], int) or not 1 <= timeouts[name] <= 1952257:
                raise ValueError("Invalid default timeout; must also allow the dynamic margin")
        if args.validate_only or args.check_connection:
            audit = dict(status="validated_no_workload_execution", protocol=PROTOCOL,
                    experiments={name:len(qs) for name,qs in data.items()},
                    default_timeout_seconds={name:timeouts[name] for name in names}, csv_columns=len(HEADER))
            if args.check_connection:
                audit["connections"] = check_connections(names,config)
            print(json.dumps(audit, indent=2))
            return 0
        options = {name:connection_options(name,config) for name in names}
        results = ensure_results(root)
        with runner_lock(results):
            safe_events = True
            for name in names:
                append_json(results / "launcher_events.jsonl", dict(event="experiment_start", experiment=name, at=now()))
                run_experiment(root,name,data[name],timeouts[name]*1000,options[name])
                append_json(results / "launcher_events.jsonl", dict(event="experiment_complete", experiment=name, at=now()))
            archive_results(root, names)
        return 0
    except KeyboardInterrupt:
        if safe_events:
            append_json(results / "launcher_events.jsonl", dict(event="interrupted", at=now()))
        print("Interrupted; completed measurements were preserved.", file=sys.stderr)
        return 130
    except Exception as exc:
        if safe_events:
            append_json(results / "launcher_events.jsonl", dict(event="failure", error_type=type(exc).__name__, at=now()))
        if isinstance(exc, (ValueError, RuntimeError, ImportError)):
            message = str(exc)
        elif isinstance(exc, OSError):
            message = "Cannot write or replace output files; check disk space, permissions and file locks."
        else:
            message = "Check that PostgreSQL is running and that connection settings/credentials are correct."
        print(f"Run failed ({type(exc).__name__}): {message}", file=sys.stderr)
        return 1
