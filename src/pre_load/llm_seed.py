"""One optional LLM supplement after CSV preprocessing, before training."""

import asyncio
import json
import math
import os
from pathlib import Path
import re
import tempfile


_PROGRESS_INTERVAL_SECONDS = 10.0


def _online_settings():
    from RL.config import CONFIGURATION_PATH

    values = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8-sig"))
    settings = dict(values.get("online_model") or {})
    key = str(settings.get("api_key") or "").strip()
    key = key or os.environ.get(str(settings.get("api_key_env") or "DASHSCOPE_API_KEY"), "").strip()
    if not key or not str(settings.get("base_url") or "").strip() or not settings.get("name"):
        raise ValueError("Online model endpoint, name and API key are required.")
    if type(settings.get("stream", True)) is not bool:
        raise ValueError("online_model.stream must be a boolean.")
    if "enable_thinking" in settings and type(settings["enable_thinking"]) is not bool:
        raise ValueError("online_model.enable_thinking must be a boolean.")
    settings["max_tokens"] = int(settings.get("max_tokens", 2048))
    if settings["max_tokens"] < 1:
        raise ValueError("online_model.max_tokens must be positive.")
    return settings, key


def _create_client(settings, key):
    import httpx
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        api_key=key,
        base_url=settings["base_url"],
        timeout=httpx.Timeout(
            connect=30, read=max(180, float(settings.get("timeout_seconds", 180))),
            write=30, pool=30,
        ),
        max_retries=0,
    )


def _request_budget(settings, context):
    count = max(1, len(context.nodes)) * 2
    # Budget the initial supplement locally; independent LLM experiments keep
    # their configured token limits and timeouts.
    tokens = math.ceil(max(settings["max_tokens"], count * 48 + 256) / 1024) * 1024
    seconds = max(600, math.ceil((60 + tokens / 25) / 60) * 60)
    return count, tokens, seconds


async def _request(client, settings, context, count, tokens):
    prompt = (
        f"Generate {count} distinct short, simple WHERE predicates for {context.db_type}: "
        "two candidate predicates for EVERY supplied Stage-2 node. "
        "First cover every node in list order with one predicate, then make a second pass "
        "with a different predicate for each node. Do not stop at a small sample: "
        f"provide at least {len(context.nodes)} predicates, covering every node once. "
        "Each predicate must actually use its corresponding operator or function; "
        "for terminal nodes, use a real column or literal of the corresponding type. "
        "Wrap non-boolean expressions in a simple comparison or IS NULL test as needed. "
        "Use only the supplied functions, safe small literal values and shallow expressions. "
        "COL_* and VALUE_* nodes are type descriptions, not SQL identifiers. "
        "Use the actual column names below, unqualified and correctly quoted for the dialect. "
        "Return raw SQL predicates, not SELECT statements or typed DSL wrappers. "
        "No subqueries, casts, comments, semicolons, assignments, user-defined functions, "
        "schema changes, data changes or session changes. "
        "Avoid division by zero, invalid domains and large string lengths. "
        "A predicate returning no rows is fine: only executability matters. "
        "Return ONLY one complete JSON object {\"predicates\":[\"predicate\",...]} "
        "within the output token budget, with no Markdown.\n"
        + json.dumps({"table": context.table_sql, "columns": context.columns,
                      "stage2_nodes": context.nodes}, ensure_ascii=False)
    )
    options = dict(
        model=settings["name"],
        messages=[{"role": "user", "content": prompt}],
        temperature=float(settings.get("temperature", 0)),
        max_tokens=tokens,
        stream=settings.get("stream", True),
    )
    if "enable_thinking" in settings:
        options["extra_body"] = {"enable_thinking": settings["enable_thinking"]}
    # This is the only API request; receiving a response establishes the connection.
    return await client.chat.completions.create(**options)


def _read_predicates(text):
    """Recover complete JSON strings, including from a truncated final array."""
    text = text.strip()
    if text.startswith("```"):
        text = text.partition("\n")[2].strip()
        if text.endswith("```"):
            text = text[:-3].rstrip()
    try:
        data = json.loads(text)
        predicates = data.get("predicates") if isinstance(data, dict) else data
    except json.JSONDecodeError:
        head = re.match(r'\s*(?:\{\s*"predicates"\s*:\s*)?\[', text)
        if head is None:
            raise ValueError("Expected a JSON predicate list.")
        predicates, offset = [], head.end()
        decoder = json.JSONDecoder()
        while offset < len(text):
            while offset < len(text) and text[offset].isspace():
                offset += 1
            try:
                value, offset = decoder.raw_decode(text, offset)
            except json.JSONDecodeError:
                break  # Never guess how an unfinished predicate should end.
            predicates.append(value)
            while offset < len(text) and text[offset].isspace():
                offset += 1
            if offset >= len(text) or text[offset] != ",":
                break
            offset += 1
    if not isinstance(predicates, list):
        raise ValueError("Expected a JSON predicate list.")
    return list(dict.fromkeys(p.strip() for p in predicates if isinstance(p, str) and p.strip()))


def _print_generation_progress(text, target):
    try:
        completed = len(_read_predicates(text))
    except ValueError:
        completed = 0
    fraction = min(completed / max(1, target), 1.0)
    filled = int(fraction * 40)
    bar = "#" * filled + "-" * (40 - filled)
    print(
        f"[LLM] Predicate generation: [{bar}] {fraction * 100:.2f}% "
        f"({completed}/{target})",
        flush=True,
    )


async def _generate(settings, key, context):
    count, tokens, seconds = _request_budget(settings, context)
    try:
        client = _create_client(settings, key)
    except Exception as exc:
        raise ConnectionError("Unable to create the LLM client.") from exc
    response, connected, interrupted = None, False, None
    pieces = []
    progress_task = None

    async def report_progress():
        while True:
            await asyncio.sleep(_PROGRESS_INTERVAL_SECONDS)
            _print_generation_progress("".join(pieces), count)

    async def receive():
        nonlocal response, connected, progress_task
        response = await _request(client, settings, context, count, tokens)
        connected = True
        print("LLM connection successful!", flush=True)
        print("Waiting for the LLM to generate predicates...", flush=True)
        _print_generation_progress("", count)
        progress_task = asyncio.create_task(report_progress())
        if settings.get("stream", True):
            size = 0
            async for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                pieces.append(content[:max(0, 1_000_000 - size)])
                size += len(content)
                if size > 1_000_000:
                    raise ValueError("LLM output exceeds the response limit.")
        else:
            pieces.append(response.choices[0].message.content or "")

    try:
        # Unlike an HTTP idle timeout, this also stops a response that keeps
        # trickling data indefinitely. Completed strings survive cancellation.
        await asyncio.wait_for(receive(), timeout=seconds)
    except Exception as exc:
        if not connected:
            raise ConnectionError("The LLM request failed.") from exc
        interrupted = exc
    finally:
        if progress_task is not None:
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass
        if response is not None and settings.get("stream", True):
            try:
                await response.close()
            except Exception:
                pass
        try:
            await client.close()
        except Exception:
            pass
        if connected:
            _print_generation_progress("".join(pieces), count)
    predicates = _read_predicates("".join(pieces))
    if interrupted is not None and not predicates:
        raise ValueError("No complete predicates were received.") from interrupted
    return predicates


def _write_supplement(path, templates):
    """Publish atomically; preserve every byte of the original CSV-derived file."""
    path = Path(path)
    original = path.read_bytes()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                         prefix=".llm-seed-", delete=False) as output:
            temporary = Path(output.name)
            output.write(original)
            if original and not original.endswith(b"\n"):
                output.write(b"\n")
            output.write(("\n".join(templates) + "\n").encode("utf-8"))
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def supplement_predicates(output_file, original_templates):
    print("Entering the LLM predicate generation stage.", flush=True)
    context = None

    def fallback(message):
        print(message, flush=True)
        print("Starting the experiment with default presets!", flush=True)
        return original_templates

    try:
        try:
            settings, key = _online_settings()
        except Exception:
            return fallback("LLM connection failed!")

        try:
            from .llm_seed_validation import open_context, validate_predicate, predicate_template
            context = open_context()
        except Exception:
            return fallback("Unable to prepare database metadata for LLM generation.")

        try:
            predicates = asyncio.run(_generate(settings, key, context))
        except ConnectionError:
            return fallback("LLM connection failed!")
        except Exception:
            return fallback("LLM predicate generation failed!")

        print("LLM generation finished. Validating predicates in the database...", flush=True)
        valid = []
        for predicate in predicates:
            try:
                parsed = validate_predicate(predicate, context)
            except Exception:
                continue
            if parsed is not None:
                valid.append(parsed)

        print("Validation finished. Converting and writing predicates...", flush=True)
        seen, additions = set(original_templates), []
        for parsed in valid:
            try:
                template = predicate_template(parsed, context)
            except Exception:
                continue
            if template and template not in seen:
                seen.add(template)
                additions.append(template)
        if additions:
            _write_supplement(output_file, additions)
        return list(original_templates) + additions
    finally:
        if context is not None:
            try:
                context.executor.close()
            except Exception:
                pass
        print("LLM generation stage finished.", flush=True)
