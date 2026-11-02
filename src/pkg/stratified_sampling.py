"""仅限时枚举/随机使用的分层调度；谓词解码、渲染仍复用原采样器。"""

import os
import time
from contextlib import ExitStack

from .stage3_runtime import Stage3DeadlineReached, encode_count


def node_level(node_id):
    # 展开器的 L0 是完整树，层号越大，剩余子树越浅。
    return int(str(node_id).rsplit("::L", 1)[1])


def sample_stratified_file(sampler, dag_data, *, output_dir, output_name,
                           deadline, compressed, compression_level, buffer_bytes,
                           progress_callback=None):
    from .sampler import _new_sampling_timings, _open_predicate_writer

    if deadline is None:
        raise ValueError("Stratified timed sampling requires a deadline")
    nodes = dag_data["nodes"]
    timings = _new_sampling_timings()
    sampler.last_sampling_timings = timings
    for timer, prepare in (
        ("successor_ordering_seconds", sampler._order_successors_in_place),
        ("decode_cache_preparation_seconds", sampler._prepare_decode_cache),
    ):
        started = time.monotonic()
        try:
            prepare(nodes, deadline=deadline)
        finally:
            timings[timer] += time.monotonic() - started

    grouped = {}
    for node_id, node in nodes.items():
        if (node.get("return_type") == "T_BOOL" and not node.get("is_terminal")
                and node.get("slots") and int(node.get("count", 0)) > 0):
            grouped.setdefault(node_level(node_id), []).append(node_id)
    states = []
    catalog = {}
    generated = 0
    last_progress = time.monotonic()
    # 一秒时间片只影响调度；每层保留自己的 root/index，绝不重头枚举。
    enumeration_slice_seconds = 1.0
    paths = []
    try:
        with ExitStack() as stack:
            for level in sorted(grouped, reverse=True):
                suffix = ".txt.gz" if compressed else ".txt"
                name = output_name[:-len(suffix)] + f"-L{level}" + suffix
                path = os.path.join(output_dir, name)
                paths.append(path)
                writer = stack.enter_context(_open_predicate_writer(
                    path + ".partial", compressed=compressed,
                    compression_level=compression_level, buffer_bytes=buffer_bytes,
                ))
                roots = grouped[level]
                specs, random_info = (sampler._prepare_local_random_roots(roots, nodes)
                                      if sampler.localized_random else (None, None))
                spans = []
                state = dict(level=level, roots=roots, root_pos=0, index=0,
                             generated=0, spent=0.0, render=0.0, write=0.0,
                             bytes=0, characters=0, path=path, writer=writer,
                             specs=specs, random_info=random_info, spans=spans,
                             primary=None, exhausted=False)
                state["tracker"] = sampler._new_component_feedback_tracker(nodes, spans, catalog)
                states.append(state)
                print(f"  [Stratified] L{level}: roots={len(roots)}, "
                      f"selected={len(specs) if specs is not None else 'enumeration'}")

            active = list(states)
            allocated = max(0.0, deadline - time.monotonic()) / max(1, len(active))
            for state in states:
                state["initial_budget"] = allocated if not sampler.localized_random else None
            print(f"  [Stratified] layers={len(states)}, remaining_generation_seconds="
                  f"{max(0.0, deadline - time.monotonic()):.2f}, "
                  f"initial_seconds_per_layer={allocated:.2f}")
            turn = 0
            while active and time.monotonic() < deadline:
                if sampler.localized_random:
                    # 每层同样大小的生成块；不足一块的最后一层保留有效前缀。
                    state = active[turn % len(active)]
                    turn += 1
                    spec = sampler.local_random_rng.choice(state["specs"])
                    root_id = spec["root_id"]
                    state["random_info"]["blocks_started"] += 1
                else:
                    # 总是补给已使用时间最少的层；耗尽层自动退出，预算自动重分配。
                    state = min(active, key=lambda item: (item["spent"], -item["level"]))
                started = time.monotonic()
                slice_end = min(deadline, started + enumeration_slice_seconds)
                block_count = 0
                try:
                    while time.monotonic() < deadline:
                        if sampler.localized_random:
                            if block_count >= sampler.local_random_root_block_size:
                                break
                            index = sampler.local_random_rng.randrange(spec["window_start"], spec["window_end"])
                            state["random_info"]["draws_attempted"] += 1
                        else:
                            if time.monotonic() >= slice_end:
                                break
                            if state["root_pos"] >= len(state["roots"]):
                                state["exhausted"] = True
                                break
                            root_id = state["roots"][state["root_pos"]]
                            index = state["index"]
                            if index >= int(nodes[root_id]["count"]):
                                state["root_pos"] += 1
                                state["index"] = 0
                                continue
                        next_case = state["generated"] + 1
                        trace = sampler._should_trace_component_feedback(state["tracker"], next_case)
                        render_started = time.monotonic()
                        try:
                            sql, edges = sampler._generate_predicate_with_trace(
                                root_id, index, nodes, collect_trace=trace, deadline=deadline)
                        finally:
                            elapsed = time.monotonic() - render_started
                            state["render"] += elapsed
                            timings["predicate_traversal_render_seconds"] += elapsed
                            timings["predicate_traversal_attempts"] += 1
                        block_count += 1
                        if not sampler.localized_random:
                            state["index"] += 1
                        if not sql or "ERROR" in sql:
                            continue
                        line = sql + "\n"
                        encoded = line.encode("utf-8")
                        write_started = time.monotonic()
                        state["writer"].write(encoded)
                        elapsed = time.monotonic() - write_started
                        timings["predicate_output_write_seconds"] += elapsed
                        timings["predicate_output_write_calls"] += 1
                        state["write"] += elapsed
                        state["bytes"] += len(encoded)
                        state["characters"] += len(line)
                        state["generated"] += 1
                        generated += 1
                        primary = state["primary"]
                        if primary is None or state.get("last_root") != root_id:
                            key, detail = sampler._runtime_signature_identity(root_id, nodes[root_id])
                            catalog.setdefault(key, detail)
                            primary = dict(start_case_idx=next_case, end_case_idx=next_case, signature_key=key)
                            state["spans"].append(primary)
                            state["primary"] = primary
                            state["last_root"] = root_id
                        primary["end_case_idx"] = next_case
                        if trace:
                            sampler._record_component_feedback_trace(state["tracker"], next_case, edges)
                        if generated % 1000 == 0 and time.monotonic() - last_progress >= 5.0:
                            last_progress = time.monotonic()
                            if progress_callback is not None:
                                progress_callback(generated)
                            print(f"  [Stratified Progress] generated={generated}, layer=L{state['level']}")
                except Stage3DeadlineReached:
                    break
                finally:
                    state["spent"] += time.monotonic() - started
                if state["exhausted"]:
                    active.remove(state)
            for state in states:
                sampler._flush_component_feedback_block(state["tracker"], state["generated"])
        for path in paths:
            os.replace(path + ".partial", path)
    except BaseException:
        # 仅清理本次创建的未提交文件，不碰历史实验的正式输出。
        for path in paths:
            if os.path.exists(path + ".partial"):
                os.remove(path + ".partial")
        raise

    outputs = []
    for state in states:
        info = dict(dag_level=state["level"], generated=state["generated"],
                    generation_active_seconds=state["spent"],
                    predicate_traversal_render_seconds=state["render"],
                    predicate_output_write_seconds=state["write"],
                    initial_generation_budget_seconds=state["initial_budget"],
                    exhausted=state["exhausted"], available_root_count=len(state["roots"]))
        if state["random_info"] is not None:
            state["random_info"]["generated_count"] = state["generated"]
            info["random_sampling"] = state["random_info"]
        elif state["root_pos"] < len(state["roots"]):
            info["stop_root_id"] = state["roots"][state["root_pos"]]
            info["next_index"] = encode_count(state["index"])
        outputs.append(dict(path=state["path"], generated=state["generated"],
                            text_bytes=state["bytes"], text_characters=state["characters"],
                            signature_spans=state["spans"], layer=info))
    return dict(outputs=outputs, signature_catalog=catalog,
                round_complete=all(state["exhausted"] for state in states))
