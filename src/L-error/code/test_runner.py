import sys
sys.dont_write_bytecode = True
from pathlib import Path
import csv
import json
import tempfile
import unittest
import contextlib
import io
import os
import runpy
import shutil
from types import SimpleNamespace
from unittest.mock import patch

import runner

ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "results"
TEST_ROOT.mkdir(parents=True, exist_ok=True)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEST_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / "results"
        self.out.mkdir()
        self.connection = SimpleNamespace(closed=False, broken=False)
        self.query = dict(query_id="example", sql="SELECT 1", sql_sha256="test")
        self.calls = []

    def measure(self, connection, sql, hint, repetition, order, timeout):
        self.calls.append((hint.hint_id, repetition, timeout))
        value = {1: [12000, 10000, 11000][repetition-1], 2: 2000, 3: 0}.get(hint.hint_id, 5000)
        status = "timeout" if hint.hint_id == 3 else "success"
        return dict(hint_id=hint.hint_id, repetition=repetition, execution_order=order,
            attempted=True, success=status=="success", status=status, latency_ms=value,
            explain=[{"Plan":{}, "Execution Time":value, "Planning Time":999999}] if status=="success" else None)

    def test_sequential_hints_and_dynamic_timeout(self):
        row = runner.run_query(self.connection, self.query, 600000, self.out, self.measure)
        self.assertEqual([x[0] for x in self.calls], [1,1,1]+list(range(2,50)))
        self.assertEqual([x[2] for x in self.calls[:3]], [600000]*3)
        self.assertEqual([x[2] for x in self.calls[3:6]], [11000,3000,3000])
        self.assertEqual(self.calls[-1][2], 3000)
        self.assertEqual(len(row), 52)
        self.assertEqual(row[1:4], ["10000.00","2000.00",runner.TIMEOUT_LABEL])
        self.assertEqual(row[-2:], ["8000.00","5.00"])
        records = [json.loads(line) for line in (self.out/"measurements.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 51)
        self.assertEqual(records[4]["latency_ms"], None)
        self.assertEqual(records[0]["planning_time_ms"], 999999)
        self.assertEqual(len(list((self.out/"plans"/"example").glob("*.gz"))), 50)

    def test_all_default_timeout_skips_hints_and_never_fills_latency(self):
        def timeout(*args):
            self.calls.append(args)
            return dict(hint_id=1, repetition=args[3], attempted=True, success=False,
                        status="timeout", latency_ms=600000, explain=None)
        row=runner.run_query(self.connection,self.query,600000,self.out,timeout)
        self.assertEqual(len(self.calls),3)
        self.assertEqual(row[1],runner.TIMEOUT_LABEL)
        self.assertEqual(set(row[2:50]),{"not_executed"})
        self.assertEqual(row[-2:],["NA","NA"])

    def test_default_mixed_success_uses_min_success(self):
        rs=[dict(hint_id=1,success=True,latency_ms=7,status="success",attempted=True),
            dict(hint_id=1,success=False,latency_ms=None,status="timeout",attempted=True),
            dict(hint_id=1,success=True,latency_ms=8,status="success",attempted=True)]
        row,summary=runner.summarize("x",rs)
        self.assertEqual(row[1],"7.00")
        self.assertEqual(summary["l_error_ms"],0)
        self.assertEqual(row[-1],"1.00")

    def test_zero_and_unrounded_calculation(self):
        row,s=runner.summarize("x",[dict(hint_id=1,success=True,latency_ms=0,status="success")])
        self.assertEqual(row[1],"0.00")
        self.assertEqual(row[-2:],["0.00","NA"])
        self.assertEqual(runner.dynamic_timeout_ms(0),1000)
        self.assertEqual(runner.dynamic_timeout_ms(15000.01),16501)
        self.assertEqual(runner.dynamic_timeout_ms(599999),659999)

    def test_only_selected_results_are_cleared(self):
        for name in runner.EXPERIMENTS:
            (self.out/(name+"_results")).mkdir()
            (self.out/(name+"_results")/"sentinel.txt").write_text("old")
        (self.root/"Robusta-JOB").mkdir()
        (self.root/"Robusta-JOB"/"query.sql").write_text("SELECT 1")
        runner.clean_output(self.root,"Robusta-JOB")
        self.assertFalse((self.out/"Robusta-JOB_results"/"sentinel.txt").exists())
        self.assertTrue((self.out/"Robusta-TPC-H_results"/"sentinel.txt").exists())
        self.assertTrue((self.root/"Robusta-JOB"/"query.sql").exists())
        with self.assertRaises(ValueError):
            runner.clean_output(self.root,"../../outside")

    def test_linked_cleanup_is_rejected_before_deleting(self):
        selected=self.out/"Robusta-JOB_results"
        selected.mkdir()
        marker=selected/"marker"
        marker.write_text("preserve")
        with patch.object(runner,"is_link",side_effect=lambda p:p==marker):
            with self.assertRaises(ValueError):
                runner.clean_output(self.root,"Robusta-JOB")
        self.assertTrue(marker.exists())

    def test_natural_sql_order_and_baselines_not_supported(self):
        self.assertEqual(sorted(["x10","x2","x1"],key=runner.natural_key),["x1","x2","x10"])
        self.assertEqual(runner.EXPERIMENTS,("Robusta-TPC-H","Robusta-JOB"))

    def test_real_hints_match_49_unique_configurations(self):
        from core.hints import generate_configs
        hints=generate_configs()
        self.assertEqual([h.hint_id for h in hints],list(range(1,50)))
        self.assertEqual(hints[0].settings,{})
        self.assertEqual(len({tuple(sorted(h.settings.items())) for h in hints}),49)

    def test_lock_blocks_a_second_runner(self):
        with runner.runner_lock(self.out):
            with self.assertRaises(RuntimeError):
                with runner.runner_lock(self.out):
                    self.fail("Second runner acquired lock")

    def test_csv_is_exactly_52_columns(self):
        p=self.out/"sample.csv"
        row,_=runner.summarize("example",[])
        with p.open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.writer(f);w.writerow(runner.HEADER);w.writerow(row)
        with p.open(encoding="utf-8-sig",newline="") as f:
            rows=list(csv.reader(f))
        self.assertEqual([len(r) for r in rows],[52,52])
        self.assertEqual(rows[0][0],"query_id")
        self.assertEqual(rows[0][-2:],["l_error_ms","speedup"])

    def test_complete_experiment_writes_outputs_only_under_results(self):
        from core import database
        self.connection.info=SimpleNamespace(server_version=180006)
        self.connection.close=lambda:None
        options={}
        with patch.object(database,"connect_database",return_value=self.connection), \
             patch.object(database,"read_default_settings",return_value={}), \
             patch.object(database,"execute_measurement",side_effect=self.measure):
            runner.run_experiment(self.root,"Robusta-JOB",[self.query],600000,options)
        target=self.out/"Robusta-JOB_results"
        state=json.loads((target/"checkpoint.json").read_text())
        self.assertEqual(state["status"],"completed")
        with (target/"results.csv").open(encoding="utf-8-sig",newline="") as f:
            rows=list(csv.reader(f))
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[1][-2:],["8000.00","5.00"])
        self.assertEqual(list(self.root.iterdir()),[self.out])

    def test_connection_failure_preserves_existing_results(self):
        from core import database
        target=self.out/"Robusta-JOB_results"
        target.mkdir()
        marker=target/"old.csv"
        marker.write_text("preserve")
        with patch.object(database,"connect_database",side_effect=RuntimeError("offline")):
            with self.assertRaises(RuntimeError):
                runner.run_experiment(self.root,"Robusta-JOB",[self.query],600000,{})
        self.assertEqual(marker.read_text(),"preserve")

    def test_missing_results_root_and_experiment_folder_are_created(self):
        package=self.root/"fresh_package"
        package.mkdir()
        target=runner.clean_output(package,"Robusta-JOB")
        self.assertTrue((package/"results").is_dir())
        self.assertEqual(target,package/"results"/"Robusta-JOB_results")
        self.assertEqual(list(target.iterdir()),[])

    def test_nested_previous_results_are_removed(self):
        target=runner.clean_output(self.root,"Robusta-JOB")
        (target/"plans"/"old_query").mkdir(parents=True)
        (target/"plans"/"old_query"/"old.json").write_text("old")
        (target/"results.csv").write_text("old")
        runner.clean_output(self.root,"Robusta-JOB")
        self.assertEqual(list(target.iterdir()),[])

    def test_validate_only_does_not_create_results(self):
        package=self.root/"check_package"
        (package/"Robusta-JOB").mkdir(parents=True)
        (package/"Robusta-JOB"/"one.sql").write_text("SELECT 1")
        with contextlib.redirect_stdout(io.StringIO()):
            code=runner.main(package,"Robusta-JOB",{"Robusta-JOB":600},None,["--validate-only"])
        self.assertEqual(code,0)
        self.assertFalse((package/"results").exists())

    @unittest.skipUnless(os.name=="nt", "Windows sharing violation")
    def test_atomic_publication_retries_transient_windows_file_lock(self):
        original_replace=Path.replace
        calls=[]
        def temporarily_locked(path,target):
            calls.append(target)
            if len(calls)<3:
                exc=PermissionError("temporary sharing violation")
                exc.winerror=32
                raise exc
            return original_replace(path,target)
        target=self.out/"checkpoint.json"
        with patch.object(Path,"replace",temporarily_locked), patch.object(runner.time,"sleep") as sleeper:
            runner.write_json(target,{"status":"saved"})
        self.assertEqual(json.loads(target.read_text()),{"status":"saved"})
        self.assertEqual(len(calls),3)
        self.assertEqual(sleeper.call_count,2)

    def test_python_entrypoint_runs_all_106_queries_without_powershell_launcher(self):
        from core import database
        project=self.root/"portable_copy"
        package=project/"src"/"L-error"
        package.mkdir(parents=True)
        shutil.copy2(ROOT/"run.py",package/"run.py")
        shutil.copy2(ROOT/"config.py",package/"config.py")
        (project/"configuration-lerror.json").write_text(json.dumps(dict(
            db_type="postgresql", host="offline.invalid", port=5432, user="offline",
            password="offline-test", database="offline", connect_timeout=5, experiment="all",
        )),encoding="utf-8")
        shutil.copytree(ROOT/"code"/"core",package/"code"/"core")
        shutil.copy2(ROOT/"code"/"runner.py",package/"code"/"runner.py")
        for name in runner.EXPERIMENTS:
            shutil.copytree(ROOT/name,package/name)
        self.assertFalse((package/"run.ps1").exists())
        self.connection.info=SimpleNamespace(server_version=180006)
        self.connection.close=lambda:None
        def measure(*args):
            record=self.measure(*args)
            record["explain"]=None
            return record
        failures=[]
        original_run=runner.run_experiment
        def capture_failure(*args):
            try:
                return original_run(*args)
            except Exception as exc:
                failures.append(exc)
                raise
        previous_path=sys.path[:]
        previous_cwd=Path.cwd()
        try:
            os.chdir(self.root)
            with patch.object(database,"connect_database",return_value=self.connection), \
                 patch.object(database,"read_default_settings",return_value={}), \
                 patch.object(database,"execute_measurement",side_effect=measure), \
                 patch.object(runner,"run_experiment",side_effect=capture_failure), \
                 patch.object(sys,"argv",[str(package/"run.py")]), \
                 patch.dict(sys.modules), \
                 contextlib.redirect_stdout(io.StringIO()), \
                 self.assertRaises(SystemExit) as stopped:
                sys.modules.pop("config",None)
                runpy.run_path(str(package/"run.py"),run_name="__main__")
            if failures:
                raise failures[0]
            self.assertEqual(stopped.exception.code,0)
        finally:
            os.chdir(previous_cwd)
            sys.path[:]=previous_path
        self.assertEqual(len(self.calls),106*51)
        for name,expected in [("Robusta-TPC-H",47),("Robusta-JOB",59)]:
            result=package/"results"/(name+"_results")
            with (result/"results.csv").open(encoding="utf-8-sig",newline="") as f:
                rows=list(csv.reader(f))
            self.assertEqual(len(rows),expected+1)
            self.assertTrue(all(len(row)==52 for row in rows))
            self.assertEqual(json.loads((result/"checkpoint.json").read_text())["status"],"completed")
            self.assertEqual(len((result/"measurements.jsonl").read_text(encoding="utf-8").splitlines()),expected*51)
            archived=project/"data"/"L-error_results"/(name+"_results")
            self.assertEqual({p.name for p in archived.iterdir()},{"results.csv","l_error_summary.txt"})
            self.assertEqual((archived/"results.csv").read_bytes(),(result/"results.csv").read_bytes())
            report=(archived/"l_error_summary.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(report[:2],["Maximum L-error (ms): 8000.00","Average L-error (ms): 8000.00"])
            self.assertEqual(report[4:],[f"{row[0]}.sql: 8000.00" for row in rows[1:]])
        self.assertFalse((project/"data/L-error_results/launcher_events.jsonl").exists())
        self.assertFalse((project/"data/L-error_results/.runner.lock").exists())

    def make_archive_fixture(self):
        project=self.root/"archive_project"
        package=project/"src"/"L-error"
        source=package/"results"
        for name in runner.EXPERIMENTS:
            folder=source/(name+"_results")
            (folder/"plans/query").mkdir(parents=True)
            with (folder/"results.csv").open("w",encoding="utf-8-sig",newline="") as f:
                writer=csv.writer(f)
                writer.writerow(runner.HEADER)
                for query_id,value in [("q1",0.0),("q2",12.0),("q3",None)]:
                    writer.writerow([query_id]+["NA"]*49+[runner.number(value),"NA"])
                    runner.append_json(folder/"query_summaries.jsonl",dict(query_id=query_id,l_error_ms=value))
            (folder/"plans/query/plan.json.gz").write_bytes(b"plan fixture")
            (folder/"checkpoint.json").write_text('{"status":"completed"}',encoding="utf-8")
        (source/"launcher_events.jsonl").write_text('{"event":"experiment_complete"}\n',encoding="utf-8")
        return project,package

    def test_archive_replaces_only_selected_experiment_and_keeps_sources(self):
        project,package=self.make_archive_fixture()
        data=project/"data"
        archive=data/"L-error_results"
        stale=archive/"Robusta-JOB_results/old_subdirectory"
        stale.mkdir(parents=True)
        (stale/"stale.txt").write_text("old")
        (archive/"Robusta-JOB_results/old.csv").write_text("old csv")
        (archive/"launcher_events.jsonl").write_text("old events")
        other=archive/"Robusta-TPC-H_results"
        other.mkdir()
        (other/"keep.txt").write_text("previous TPC-H archive")
        (data/"readme.md").write_text("preserve README")
        (data/"LLM_results").mkdir()
        (data/"LLM_results/keep.txt").write_text("previous LLM archive")
        before={p.relative_to(package):p.read_bytes() for p in package.rglob("*") if p.is_file()}
        runner.archive_results(package,("Robusta-JOB",))
        selected=archive/"Robusta-JOB_results"
        source=package/"results/Robusta-JOB_results"
        self.assertEqual({p.name for p in selected.iterdir()},{"results.csv","l_error_summary.txt"})
        self.assertEqual((selected/"results.csv").read_bytes(),(source/"results.csv").read_bytes())
        report=(selected/"l_error_summary.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(report[:2],["Maximum L-error (ms): 12.00","Average L-error (ms): 6.00"])
        self.assertEqual(report[4:],["q1.sql: 0.00","q2.sql: 12.00","q3.sql: NA"])
        self.assertFalse((archive/"launcher_events.jsonl").exists())
        self.assertEqual((other/"keep.txt").read_text(),"previous TPC-H archive")
        self.assertEqual((data/"readme.md").read_text(),"preserve README")
        self.assertEqual((data/"LLM_results/keep.txt").read_text(),"previous LLM archive")
        self.assertEqual(before,{p.relative_to(package):p.read_bytes() for p in package.rglob("*") if p.is_file()})
        self.assertFalse(list(archive.glob(".archive-*")))

    def test_archive_creates_missing_paths_and_preserves_old_data_on_copy_failure(self):
        project,package=self.make_archive_fixture()
        archive=runner.archive_results(package,runner.EXPERIMENTS)
        before={p.relative_to(archive):p.read_bytes() for p in archive.rglob("*") if p.is_file()}
        self.assertTrue((archive/"Robusta-JOB_results/results.csv").is_file())
        self.assertTrue((archive/"Robusta-TPC-H_results/results.csv").is_file())
        with patch.object(runner.shutil,"copy2",side_effect=OSError("simulated copy failure")):
            with self.assertRaises(OSError):
                runner.archive_results(package,("Robusta-JOB",))
        self.assertEqual(before,{p.relative_to(archive):p.read_bytes() for p in archive.rglob("*") if p.is_file()})
        checkpoint=package/"results/Robusta-JOB_results/checkpoint.json"
        checkpoint.write_text('{"status":"interrupted"}',encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"incomplete"):
            runner.archive_results(package,("Robusta-JOB",))
        self.assertEqual(before,{p.relative_to(archive):p.read_bytes() for p in archive.rglob("*") if p.is_file()})
        self.assertFalse(list(archive.glob(".archive-*")))

    def test_archive_summary_uses_unrounded_values_and_handles_unavailable_errors(self):
        project,package=self.make_archive_fixture()
        summaries=package/"results/Robusta-JOB_results/query_summaries.jsonl"
        for values,expected in [
            ([1.004,1.004,1.014,None],["Maximum L-error (ms): 1.01","Average L-error (ms): 1.01"]),
            ([None,None],["Maximum L-error (ms): NA","Average L-error (ms): NA"]),
        ]:
            with self.subTest(values=values):
                summaries.write_text("".join(json.dumps(dict(query_id=f"q{i}",l_error_ms=value))+"\n"
                    for i,value in enumerate(values,1)),encoding="utf-8")
                archive=runner.archive_results(package,("Robusta-JOB",))
                report=(archive/"Robusta-JOB_results/l_error_summary.txt").read_text(encoding="utf-8").splitlines()
                self.assertEqual(report[:2],expected)
                self.assertEqual(report[4:],[f"q{i}.sql: {runner.number(value)}" for i,value in enumerate(values,1)])

    def test_invalid_summary_preserves_all_previous_archives(self):
        project,package=self.make_archive_fixture()
        archive=runner.archive_results(package,runner.EXPERIMENTS)
        before={p.relative_to(archive):p.read_bytes() for p in archive.rglob("*") if p.is_file()}
        (package/"results/Robusta-TPC-H_results/results.csv").write_text("new data")
        (package/"results/Robusta-JOB_results/query_summaries.jsonl").write_text("{invalid")
        with self.assertRaises(ValueError):
            runner.archive_results(package,runner.EXPERIMENTS)
        self.assertEqual(before,{p.relative_to(archive):p.read_bytes() for p in archive.rglob("*") if p.is_file()})
        self.assertFalse(list(archive.glob(".archive-*")))

    def test_archive_is_not_called_for_validation_or_failed_experiment(self):
        package=self.root/"no_archive"
        (package/"Robusta-JOB").mkdir(parents=True)
        (package/"Robusta-JOB/one.sql").write_text("SELECT 1")
        connection=SimpleNamespace(db_type="postgresql",host="offline.invalid",port=5432,
            user="offline",password="offline-test",database="offline",connect_timeout=5)
        with patch.object(runner,"archive_results") as archive, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(package,"Robusta-JOB",{"Robusta-JOB":600},connection,["--validate-only"]),0)
            with patch.object(runner,"check_connections",return_value={}):
                self.assertEqual(runner.main(package,"Robusta-JOB",{"Robusta-JOB":600},connection,["--check-connection"]),0)
            with patch.object(runner,"run_experiment",side_effect=RuntimeError("simulated failure")), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(runner.main(package,"Robusta-JOB",{"Robusta-JOB":600},connection,[]),1)
            archive.assert_not_called()

    def test_dataclass_config_maps_to_psycopg_without_environment_override(self):
        data=SimpleNamespace(db_type="postgresql",host="offline.invalid",port=5432,
                             user="offline",password="offline-test",database="offline",connect_timeout=5)
        with patch.dict(os.environ,{"PGHOST":"ignored.invalid","PGPORT":"9999","PGUSER":"ignored", "PGPASSWORD":"ignored"}):
            options=runner.connection_options("Robusta-JOB",data)
        self.assertEqual(options["host"],data.host)
        self.assertEqual(options["port"],data.port)
        self.assertEqual(options["dbname"],data.database)
        self.assertEqual(options["password"],data.password)
        self.assertEqual(options["options"],"-csearch_path=job,public")
        self.assertNotIn("passfile",options)

    def test_empty_connection_config_stops_before_connection_or_output_cleanup(self):
        from core import database
        empty=SimpleNamespace(db_type="postgresql",host="",port=None,user="",password="",database="",connect_timeout=5)
        (self.root/"Robusta-JOB").mkdir()
        (self.root/"Robusta-JOB"/"one.sql").write_text("SELECT 1")
        target=self.out/"Robusta-JOB_results"
        target.mkdir()
        marker=target/"old.csv"
        marker.write_text("preserve")
        error=io.StringIO()
        with patch.object(database,"connect_database") as connect, \
             patch.dict(os.environ,{"PGHOST":"ignored.invalid","PGPASSWORD":"ignored"}), \
             contextlib.redirect_stderr(error):
            result=runner.main(self.root,"Robusta-JOB",{"Robusta-JOB":600},empty,[])
        self.assertEqual(result,1)
        connect.assert_not_called()
        self.assertIn("Fill Robusta/configuration-lerror.json",error.getvalue())
        self.assertEqual(marker.read_text(),"preserve")
        self.assertEqual(list(self.out.iterdir()),[target])

    def test_root_configuration_is_read_independently_of_working_directory(self):
        project=self.root/"configured_project"
        package=project/"src"/"L-error"
        package.mkdir(parents=True)
        shutil.copy2(ROOT/"config.py",package/"config.py")
        config_path=project/"configuration-lerror.json"
        values=dict(db_type="postgresql",host="offline.invalid",port=5433,user="offline",
                    password="offline-test",database="benchmark",connect_timeout=9,
                    experiment="Robusta-JOB",_comment_experiment="Choices documented here")
        config_path.write_text(json.dumps(values),encoding="utf-8-sig")
        original=config_path.read_bytes()
        previous_cwd=Path.cwd()
        try:
            os.chdir(self.out)
            loaded=runpy.run_path(str(package/"config.py"))
        finally:
            os.chdir(previous_cwd)
        self.assertEqual(loaded["EXPERIMENT"],"Robusta-JOB")
        options=runner.connection_options("Robusta-JOB",loaded["DB_CONFIG"])
        self.assertEqual(options["host"],"offline.invalid")
        self.assertEqual(options["dbname"],"benchmark")
        self.assertEqual(options["port"],5433)
        self.assertEqual(options["connect_timeout"],9)
        self.assertEqual(options["options"],"-csearch_path=job,public")
        self.assertEqual(config_path.read_bytes(),original)
        self.assertFalse((package/"results").exists())

    def test_configuration_defaults_and_invalid_values(self):
        project=self.root/"config_validation"
        package=project/"src"/"L-error"
        package.mkdir(parents=True)
        shutil.copy2(ROOT/"config.py",package/"config.py")
        config_path=project/"configuration-lerror.json"
        config_path.write_text("{}",encoding="utf-8")
        loaded=runpy.run_path(str(package/"config.py"))
        self.assertEqual(loaded["EXPERIMENT"],"all")
        self.assertEqual(loaded["DB_CONFIG"].host,"")
        self.assertIsNone(loaded["DB_CONFIG"].port)
        for content in ('[]','{invalid',json.dumps({"experiment":"unknown"}),
                        json.dumps({"db_type":"mysql"}),json.dumps({"port":True}),
                        json.dumps({"port":65536}),json.dumps({"connect_timeout":0}),
                        json.dumps({"host":123}),json.dumps({"unexpected":1})):
            with self.subTest(content=content):
                config_path.write_text(content,encoding="utf-8")
                with self.assertRaises(ValueError):
                    runpy.run_path(str(package/"config.py"))
        config_path.unlink()
        with self.assertRaisesRegex(ValueError,"Cannot read L-error configuration"):
            runpy.run_path(str(package/"config.py"))


if __name__=="__main__":
    result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProtocolTests))
    raise SystemExit(0 if result.wasSuccessful() else 1)
