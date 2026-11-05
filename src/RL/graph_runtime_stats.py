from __future__ import annotations

"""
graph_runtime_stats.py

阶段 3：运行统计模块。

本模块只记录系统真实运行过程中已经发生的事实，不维护人工候选边全集，
不根据 MySQL / PostgreSQL 等数据库弱类型规则推导“应该存在”的边。

它记录：
1. observed_edges.json
   数据库执行成功且 parser 成功抽取到的边。

2. structure_attempts.json
   系统实际生成并尝试过的谓词结构，包括成功和失败。

3. operator_stats.json
   每个算子的出现次数、成功次数、失败次数、新边贡献等统计。

4. predicate_contribution.json
   每个成功谓词贡献过哪些边、算子、是否引入新边/新算子。

5. unknown_success_buffer.txt
   数据库执行成功，但 parser 无法解析或无法抽边的谓词。

注意：
- Correct_Collection 的标准仍然是“数据库执行成功”；
- Knowledge Graph 的标准是“数据库执行成功 + parser 成功抽边”；
- parser 失败但 SQL 成功的谓词仍然是正确谓词，只是不进入知识图；
- 这里不替 RL 生成谓词，也不修复谓词。
"""

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set
from contextlib import contextmanager
import hashlib
import json
import os
import tempfile
import time

from RL.predicate_structure import ExtractedEdge, ParsedPredicateResult
from RL.joint_signature_evidence import JointSignatureEvidenceStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATS_DIR = PROJECT_ROOT / "output" / "graph_runtime_stats"


# ---------------------------------------------------------------------------
# 基础工具函数
# ---------------------------------------------------------------------------

def _now_ts() -> float:
    """返回当前 Unix 时间戳。"""
    return float(time.time())


def _predicate_key(predicate: str) -> str:
    """
    为谓词生成稳定 key。

    不直接把谓词当 JSON key 的原因：
    1. 谓词可能很长；
    2. 谓词中可能包含引号、换行等字符；
    3. hash key 更适合后续做去重和索引。
    """
    text = "" if predicate is None else str(predicate)
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _edge_key_from_parts(source_node_id: str, slot_id: int, target_node_id: str) -> str:
    """生成结构边唯一 key。"""
    return f"{source_node_id}::slot{int(slot_id)}->{target_node_id}"


def edge_to_key(edge: ExtractedEdge) -> str:
    """将 ExtractedEdge 转成唯一 key。"""
    return _edge_key_from_parts(
        source_node_id=edge.source_node_id,
        slot_id=edge.slot_id,
        target_node_id=edge.target_node_id,
    )


def _json_default(obj: Any) -> Any:
    """
    json.dump 的兜底序列化函数。

    ExtractedEdge 是 dataclass，可以直接转 dict。
    其他不可序列化对象统一转字符串，避免统计模块影响主流程。
    """
    if is_dataclass(obj):
        return asdict(obj)
    return str(obj)


@contextmanager
def _file_lock(target_path: Path, timeout_seconds: float = 10.0, poll_interval: float = 0.05):
    """
    简单跨平台文件锁。

    这里不用第三方库，避免引入额外依赖。
    通过创建 .lock 文件实现互斥：
    - 创建成功：获得锁；
    - 文件已存在：等待；
    - 超时：抛 TimeoutError。

    注意：
    多进程同时写统计文件时，必须避免 JSON 被写坏。
    """
    target_path = Path(target_path)
    lock_path = target_path.with_name(target_path.name + ".lock")

    start = time.time()
    fd: Optional[int] = None

    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            break
        except FileExistsError:
            if time.time() - start > timeout_seconds:
                raise TimeoutError(f"等待文件锁超时: {lock_path}")
            time.sleep(poll_interval)

    try:
        yield
    finally:
        try:
            if fd is not None:
                os.close(fd)
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def _load_json(path: Path, default: Any) -> Any:
    """安全读取 JSON。文件不存在或内容为空时返回 default。"""
    path = Path(path)

    if not path.exists():
        return default

    try:
        if path.stat().st_size == 0:
            return default
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # 统计文件不应该中断 RL 主流程。
        # 如果 JSON 损坏，先返回 default，后续可人工检查备份。
        return default


def _atomic_write_json(path: Path, data: Any) -> None:
    """
    原子写 JSON。

    先写临时文件，再 os.replace 覆盖目标文件，避免写到一半进程退出导致文件损坏。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=str(path.parent),
        suffix=".tmp",
    ) as tmp:
        json.dump(data, tmp, ensure_ascii=False, indent=2, default=_json_default)
        tmp.write("\n")
        tmp_path = Path(tmp.name)

    os.replace(str(tmp_path), str(path))


def _append_line_locked(path: Path, line: str) -> None:
    """加锁追加一行文本。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with _file_lock(path):
        with path.open("a", encoding="utf-8") as f:
            f.write(line.rstrip("\n") + "\n")


def _append_lines_locked(path: Path, lines: List[str]) -> None:
    """加锁追加多行文本。"""
    if not lines:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with _file_lock(path):
        with path.open("a", encoding="utf-8") as f:
            for line in lines:
                f.write(str(line).rstrip("\n") + "\n")


def _parse_result_edges(parse_result: Optional[ParsedPredicateResult]) -> List[ExtractedEdge]:
    """从 ParsedPredicateResult 中安全取出边列表。"""
    if parse_result is None:
        return []
    return list(getattr(parse_result, "edges", []) or [])


def _parse_result_operators(parse_result: Optional[ParsedPredicateResult]) -> Set[str]:
    """从 ParsedPredicateResult 中安全取出算子集合。"""
    if parse_result is None:
        return set()
    return set(getattr(parse_result, "operators", set()) or set())


def _parse_result_node_ids(parse_result: Optional[ParsedPredicateResult]) -> Set[str]:
    """从 ParsedPredicateResult 中安全取出节点集合。"""
    if parse_result is None:
        return set()
    return set(getattr(parse_result, "node_ids", set()) or set())


def _parse_success(parse_result: Optional[ParsedPredicateResult]) -> bool:
    """安全判断 parser 是否成功。"""
    if parse_result is None:
        return False
    return bool(getattr(parse_result, "success", False))


def _coverage_profile() -> Dict[str, Any]:
    try:
        from RL.config import load_db_coverage_profile
        return load_db_coverage_profile()
    except Exception:
        return {}


def _coverage_base_node_ids() -> Set[str]:
    profile = _coverage_profile()
    return {str(x) for x in profile.get("base_node_ids", [])}


def _coverage_mode() -> str:
    try:
        from RL.config import RL_ENV_CONFIG
        return str(getattr(RL_ENV_CONFIG, "coverage_mode", "unrestricted")).lower()
    except Exception:
        return "unrestricted"


def _is_fixed_base_parse_result(parse_result: Optional[ParsedPredicateResult]) -> bool:
    if parse_result is None or not _parse_success(parse_result):
        return False

    base_nodes = _coverage_base_node_ids()
    if not base_nodes:
        return False

    node_ids = _parse_result_node_ids(parse_result)
    if not node_ids:
        return False

    return node_ids.issubset(base_nodes)


def _coverage_split(parse_result: Optional[ParsedPredicateResult], graph_new_edges: int, graph_new_nodes: int) -> Dict[str, Any]:
    fixed_base_structure = _is_fixed_base_parse_result(parse_result)

    if int(graph_new_edges) <= 0:
        fixed_base_new_edges = 0
        frontier_new_edges = 0
    elif fixed_base_structure:
        fixed_base_new_edges = int(graph_new_edges)
        frontier_new_edges = 0
    else:
        fixed_base_new_edges = 0
        frontier_new_edges = int(graph_new_edges)

    return {
        "coverage_mode": _coverage_mode(),
        "fixed_base_structure": bool(fixed_base_structure),
        "fixed_base_new_edges": int(fixed_base_new_edges),
        "frontier_new_edges": int(frontier_new_edges),
        "frontier_new_nodes": int(graph_new_nodes) if not fixed_base_structure else 0,
    }


# ---------------------------------------------------------------------------
# 主类：GraphRuntimeStats
# ---------------------------------------------------------------------------

class GraphRuntimeStats:
    """
    运行统计管理器。

    建议用法：
        stats = GraphRuntimeStats()
        stats.record_sql_success(...)
        stats.record_sql_failure(...)
        stats.record_placeholder(...)

    后续阶段中：
    - env_sql_dependency.py 可以用它记录 step 级尝试；
    - pkg_manager.py 可以用它记录全局知识图真实接收的 Correct 谓词；
    """

    def __init__(self, stats_dir: Optional[Path] = None) -> None:
        self.stats_dir = Path(stats_dir) if stats_dir is not None else DEFAULT_STATS_DIR
        self.stats_dir.mkdir(parents=True, exist_ok=True)

        self.observed_edges_path = self.stats_dir / "observed_edges.json"
        self.structure_attempts_path = self.stats_dir / "structure_attempts.json"
        self.operator_stats_path = self.stats_dir / "operator_stats.json"
        self.predicate_contribution_path = self.stats_dir / "predicate_contribution.json"
        self.unknown_success_buffer_path = self.stats_dir / "unknown_success_buffer.txt"
        self.joint_signature_evidence = JointSignatureEvidenceStore(self.stats_dir)

    # ------------------------------------------------------------------
    # 读取接口
    # ------------------------------------------------------------------

    def load_observed_edges(self) -> Dict[str, Any]:
        return _load_json(self.observed_edges_path, {})

    def load_structure_attempts(self) -> Dict[str, Any]:
        return _load_json(self.structure_attempts_path, {})

    def load_operator_stats(self) -> Dict[str, Any]:
        return _load_json(self.operator_stats_path, {})

    def load_predicate_contribution(self) -> Dict[str, Any]:
        return _load_json(self.predicate_contribution_path, {})

    # ------------------------------------------------------------------
    # 写入接口
    # ------------------------------------------------------------------

    def record_sql_success_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not records:
            return []

        prepared: List[Dict[str, Any]] = []
        for record in records:
            pred = "" if record.get("predicate") is None else str(record.get("predicate"))
            parse_result = record.get("parse_result")
            parse_ok = _parse_success(parse_result)
            edges = _parse_result_edges(parse_result)
            operators = _parse_result_operators(parse_result)
            node_ids = _parse_result_node_ids(parse_result)
            edge_keys = [edge_to_key(edge) for edge in edges]
            coverage_info = _coverage_split(
                parse_result=parse_result,
                graph_new_edges=int(record.get("graph_new_edges", 0) or 0),
                graph_new_nodes=int(record.get("graph_new_nodes", 0) or 0),
            )

            prepared.append({
                "predicate": pred,
                "parse_result": parse_result,
                "step": record.get("step"),
                "source": record.get("source", "unknown"),
                "graph_new_nodes": int(record.get("graph_new_nodes", 0) or 0),
                "graph_new_edges": int(record.get("graph_new_edges", 0) or 0),
                "coverage_mode": coverage_info["coverage_mode"],
                "fixed_base_structure": coverage_info["fixed_base_structure"],
                "fixed_base_new_edges": coverage_info["fixed_base_new_edges"],
                "frontier_new_edges": coverage_info["frontier_new_edges"],
                "frontier_new_nodes": coverage_info["frontier_new_nodes"],
                "ts": _now_ts(),
                "pred_key": _predicate_key(pred),
                "parse_ok": parse_ok,
                "edges": edges,
                "operators": operators,
                "node_ids": node_ids,
                "edge_keys": edge_keys,
                "normalized_predicate": getattr(parse_result, "normalized_predicate", pred) if parse_result else pred,
                "abstract_predicate": getattr(parse_result, "abstract_predicate", "") if parse_result else "",
                "parse_error": getattr(parse_result, "error", None) if parse_result else "parse_result is None",
                "new_edge_keys": [],
                "new_operator_ids": [],
            })

        # Complete sibling combinations are separate from legacy per-edge
        # statistics and are flushed only after a large in-memory batch.
        try:
            self.joint_signature_evidence.observe_success_batch(
                item["parse_result"] for item in prepared
            )
        except Exception as exc:
            print(f"[RuntimeStats Warning] joint signature success evidence failed: {exc}")

        with _file_lock(self.observed_edges_path):
            observed_edges = _load_json(self.observed_edges_path, {})
            for item in prepared:
                if not item["parse_ok"]:
                    continue

                for edge in item["edges"]:
                    e_key = edge_to_key(edge)
                    if e_key not in observed_edges:
                        item["new_edge_keys"].append(e_key)
                        observed_edges[e_key] = {
                            "edge_key": e_key,
                            "source_node_id": edge.source_node_id,
                            "source_token": edge.source_token,
                            "source_return_type": edge.source_return_type,
                            "slot_id": edge.slot_id,
                            "target_node_id": edge.target_node_id,
                            "target_token": edge.target_token,
                            "target_return_type": edge.target_return_type,
                            "target_node_type": edge.target_node_type,
                            "target_is_terminal": edge.target_is_terminal,
                            "first_seen_predicate": item["predicate"],
                            "first_seen_step": item["step"],
                            "first_seen_source": item["source"],
                            "first_seen_time": item["ts"],
                            "count": 0,
                            "last_seen_step": None,
                            "last_seen_source": None,
                            "last_seen_time": None,
                        }

                    observed_edges[e_key]["count"] = int(observed_edges[e_key].get("count", 0)) + 1
                    observed_edges[e_key]["last_seen_step"] = item["step"]
                    observed_edges[e_key]["last_seen_source"] = item["source"]
                    observed_edges[e_key]["last_seen_time"] = item["ts"]

            _atomic_write_json(self.observed_edges_path, observed_edges)

        unknown_lines = []
        for item in prepared:
            if item["parse_ok"]:
                continue
            payload = {
                "time": item["ts"],
                "step": item["step"],
                "source": item["source"],
                "predicate": item["predicate"],
                "abstract_predicate": item["abstract_predicate"],
                "parse_error": item["parse_error"],
            }
            unknown_lines.append(json.dumps(payload, ensure_ascii=False, default=_json_default))
        _append_lines_locked(self.unknown_success_buffer_path, unknown_lines)

        with _file_lock(self.operator_stats_path):
            op_stats = _load_json(self.operator_stats_path, {})
            for item in prepared:
                for op_id in sorted(item["operators"]):
                    if op_id not in op_stats:
                        item["new_operator_ids"].append(op_id)
                        op_stats[op_id] = {
                            "operator_id": op_id,
                            "first_seen_step": item["step"],
                            "first_seen_source": item["source"],
                            "first_seen_time": item["ts"],
                            "success_count": 0,
                            "failure_count": 0,
                            "placeholder_count": 0,
                            "parse_success_count": 0,
                            "parse_failure_count": 0,
                            "new_edge_count": 0,
                            "last_seen_step": None,
                            "last_seen_source": None,
                            "last_seen_time": None,
                        }

                    op_stats[op_id]["success_count"] = int(op_stats[op_id].get("success_count", 0)) + 1
                    op_stats[op_id]["parse_success_count"] = int(op_stats[op_id].get("parse_success_count", 0)) + int(item["parse_ok"])
                    op_stats[op_id]["last_seen_step"] = item["step"]
                    op_stats[op_id]["last_seen_source"] = item["source"]
                    op_stats[op_id]["last_seen_time"] = item["ts"]

                for edge in item["edges"]:
                    e_key = edge_to_key(edge)
                    if e_key not in item["new_edge_keys"]:
                        continue
                    op_id = edge.source_node_id
                    if op_id not in op_stats:
                        item["new_operator_ids"].append(op_id)
                        op_stats[op_id] = {
                            "operator_id": op_id,
                            "first_seen_step": item["step"],
                            "first_seen_source": item["source"],
                            "first_seen_time": item["ts"],
                            "success_count": 0,
                            "failure_count": 0,
                            "placeholder_count": 0,
                            "parse_success_count": 0,
                            "parse_failure_count": 0,
                            "new_edge_count": 0,
                            "last_seen_step": None,
                            "last_seen_source": None,
                            "last_seen_time": None,
                        }
                    op_stats[op_id]["new_edge_count"] = int(op_stats[op_id].get("new_edge_count", 0)) + 1

            _atomic_write_json(self.operator_stats_path, op_stats)

        with _file_lock(self.predicate_contribution_path):
            contrib = _load_json(self.predicate_contribution_path, {})
            for item in prepared:
                pred_key = item["pred_key"]
                if pred_key not in contrib:
                    contrib[pred_key] = {
                        "predicate_key": pred_key,
                        "predicate": item["predicate"],
                        "normalized_predicate": item["normalized_predicate"],
                        "abstract_predicate": item["abstract_predicate"],
                        "first_seen_step": item["step"],
                        "first_seen_source": item["source"],
                        "first_seen_time": item["ts"],
                        "success_count": 0,
                        "failure_count": 0,
                        "parse_success": False,
                        "parse_error": None,
                        "edges": [],
                        "operators": [],
                        "node_ids": [],
                        "new_edges": [],
                        "new_operators": [],
                        "graph_new_nodes": 0,
                        "graph_new_edges": 0,
                        "coverage_mode": item["coverage_mode"],
                        "fixed_base_structure": bool(item["fixed_base_structure"]),
                        "fixed_base_new_edges": 0,
                        "frontier_new_edges": 0,
                        "frontier_new_nodes": 0,
                        "added_to_predicate": False,
                        "last_seen_step": None,
                        "last_seen_source": None,
                        "last_seen_time": None,
                    }

                contrib_item = contrib[pred_key]
                contrib_item["success_count"] = int(contrib_item.get("success_count", 0)) + 1
                contrib_item["parse_success"] = bool(item["parse_ok"])
                contrib_item["parse_error"] = item["parse_error"]
                contrib_item["normalized_predicate"] = item["normalized_predicate"]
                contrib_item["abstract_predicate"] = item["abstract_predicate"]
                contrib_item["edges"] = sorted(set(contrib_item.get("edges", []) + item["edge_keys"]))
                contrib_item["operators"] = sorted(set(contrib_item.get("operators", []) + list(item["operators"])))
                contrib_item["node_ids"] = sorted(set(contrib_item.get("node_ids", []) + list(item["node_ids"])))
                contrib_item["new_edges"] = sorted(set(contrib_item.get("new_edges", []) + item["new_edge_keys"]))
                contrib_item["new_operators"] = sorted(set(contrib_item.get("new_operators", []) + item["new_operator_ids"]))
                contrib_item["graph_new_nodes"] = int(contrib_item.get("graph_new_nodes", 0)) + item["graph_new_nodes"]
                contrib_item["graph_new_edges"] = int(contrib_item.get("graph_new_edges", 0)) + item["graph_new_edges"]
                contrib_item["coverage_mode"] = item["coverage_mode"]
                contrib_item["fixed_base_structure"] = bool(item["fixed_base_structure"])
                contrib_item["fixed_base_new_edges"] = int(contrib_item.get("fixed_base_new_edges", 0)) + int(item["fixed_base_new_edges"])
                contrib_item["frontier_new_edges"] = int(contrib_item.get("frontier_new_edges", 0)) + int(item["frontier_new_edges"])
                contrib_item["frontier_new_nodes"] = int(contrib_item.get("frontier_new_nodes", 0)) + int(item["frontier_new_nodes"])
                contrib_item["last_seen_step"] = item["step"]
                contrib_item["last_seen_source"] = item["source"]
                contrib_item["last_seen_time"] = item["ts"]
                contrib[pred_key] = contrib_item

            _atomic_write_json(self.predicate_contribution_path, contrib)

        with _file_lock(self.structure_attempts_path):
            attempts = _load_json(self.structure_attempts_path, {})
            for item in prepared:
                pred_key = item["pred_key"]
                if pred_key not in attempts:
                    attempts[pred_key] = {
                        "predicate_key": pred_key,
                        "predicate": item["predicate"],
                        "normalized_predicate": item["normalized_predicate"],
                        "abstract_predicate": item["abstract_predicate"],
                        "first_seen_step": item["step"],
                        "first_seen_source": item["source"],
                        "first_seen_time": item["ts"],
                        "attempt_count": 0,
                        "success_count": 0,
                        "failure_count": 0,
                        "placeholder_count": 0,
                        "parse_success_count": 0,
                        "parse_failure_count": 0,
                        "edges": [],
                        "operators": [],
                        "last_error": None,
                        "last_seen_step": None,
                        "last_seen_source": None,
                        "last_seen_time": None,
                    }

                attempt_item = attempts[pred_key]
                attempt_item["attempt_count"] = int(attempt_item.get("attempt_count", 0)) + 1
                attempt_item["success_count"] = int(attempt_item.get("success_count", 0)) + 1
                attempt_item["failure_count"] = int(attempt_item.get("failure_count", 0))
                attempt_item["placeholder_count"] = int(attempt_item.get("placeholder_count", 0))
                if item["parse_ok"]:
                    attempt_item["parse_success_count"] = int(attempt_item.get("parse_success_count", 0)) + 1
                else:
                    attempt_item["parse_failure_count"] = int(attempt_item.get("parse_failure_count", 0)) + 1
                attempt_item["normalized_predicate"] = item["normalized_predicate"]
                attempt_item["abstract_predicate"] = item["abstract_predicate"]
                attempt_item["edges"] = sorted(set(attempt_item.get("edges", []) + item["edge_keys"]))
                attempt_item["operators"] = sorted(set(attempt_item.get("operators", []) + list(item["operators"])))
                attempt_item["last_error"] = item["parse_error"]
                attempt_item["last_seen_step"] = item["step"]
                attempt_item["last_seen_source"] = item["source"]
                attempt_item["last_seen_time"] = item["ts"]
                attempts[pred_key] = attempt_item

            _atomic_write_json(self.structure_attempts_path, attempts)

        return [
            {
                "predicate_key": item["pred_key"],
                "parse_success": item["parse_ok"],
                "edge_count": len(item["edge_keys"]),
                "new_edge_count": len(item["new_edge_keys"]),
                "operator_count": len(item["operators"]),
                "new_operator_count": len(item["new_operator_ids"]),
                "unknown_success": not item["parse_ok"],
                "coverage_mode": item["coverage_mode"],
                "fixed_base_structure": bool(item["fixed_base_structure"]),
                "fixed_base_new_edges": int(item["fixed_base_new_edges"]),
                "frontier_new_edges": int(item["frontier_new_edges"]),
                "frontier_new_nodes": int(item["frontier_new_nodes"]),
            }
            for item in prepared
        ]

    def record_sql_success(
        self,
        predicate: str,
        parse_result: Optional[ParsedPredicateResult],
        *,
        step: Optional[int] = None,
        source: str = "unknown",
        graph_new_nodes: int = 0,
        graph_new_edges: int = 0,
    ) -> Dict[str, Any]:
        """
        记录一次数据库执行成功的谓词。

        如果 parse_result.success=True：
            - 更新 observed_edges.json；
            - 更新 operator_stats.json；
            - 更新 predicate_contribution.json；
            - 更新 structure_attempts.json。

        如果 parse_result.success=False：
            - 写入 unknown_success_buffer.txt；
            - 更新 predicate_contribution.json 中的 parse_failed 记录；
            - 更新 structure_attempts.json。
        """
        fallback_summary = {
            "predicate_key": _predicate_key("" if predicate is None else str(predicate)),
            "parse_success": _parse_success(parse_result),
            "edge_count": 0,
            "new_edge_count": 0,
            "operator_count": 0,
            "new_operator_count": 0,
            "unknown_success": not _parse_success(parse_result),
        }
        summaries = self.record_sql_success_batch([
            {
                "predicate": predicate,
                "parse_result": parse_result,
                "step": step,
                "source": source,
                "graph_new_nodes": graph_new_nodes,
                "graph_new_edges": graph_new_edges,
            }
        ])
        return summaries[0] if summaries else fallback_summary

        ts = _now_ts()
        pred = "" if predicate is None else str(predicate)
        pred_key = _predicate_key(pred)

        parse_ok = _parse_success(parse_result)
        edges = _parse_result_edges(parse_result)
        operators = _parse_result_operators(parse_result)
        node_ids = _parse_result_node_ids(parse_result)

        normalized_predicate = getattr(parse_result, "normalized_predicate", pred) if parse_result else pred
        abstract_predicate = getattr(parse_result, "abstract_predicate", "") if parse_result else ""
        parse_error = getattr(parse_result, "error", None) if parse_result else "parse_result is None"

        edge_keys = [edge_to_key(edge) for edge in edges]

        new_edge_keys: List[str] = []
        new_operator_ids: List[str] = []

        with _file_lock(self.observed_edges_path):
            observed_edges = _load_json(self.observed_edges_path, {})

            if parse_ok:
                for edge in edges:
                    e_key = edge_to_key(edge)
                    if e_key not in observed_edges:
                        new_edge_keys.append(e_key)
                        observed_edges[e_key] = {
                            "edge_key": e_key,
                            "source_node_id": edge.source_node_id,
                            "source_token": edge.source_token,
                            "source_return_type": edge.source_return_type,
                            "slot_id": edge.slot_id,
                            "target_node_id": edge.target_node_id,
                            "target_token": edge.target_token,
                            "target_return_type": edge.target_return_type,
                            "target_node_type": edge.target_node_type,
                            "target_is_terminal": edge.target_is_terminal,
                            "first_seen_predicate": pred,
                            "first_seen_step": step,
                            "first_seen_source": source,
                            "first_seen_time": ts,
                            "count": 0,
                            "last_seen_step": None,
                            "last_seen_source": None,
                            "last_seen_time": None,
                        }

                    observed_edges[e_key]["count"] = int(observed_edges[e_key].get("count", 0)) + 1
                    observed_edges[e_key]["last_seen_step"] = step
                    observed_edges[e_key]["last_seen_source"] = source
                    observed_edges[e_key]["last_seen_time"] = ts

                _atomic_write_json(self.observed_edges_path, observed_edges)

        # parser 失败但 SQL 成功：进入 unknown_success_buffer
        if not parse_ok:
            self.record_unknown_success(
                predicate=pred,
                parse_error=parse_error,
                abstract_predicate=abstract_predicate,
                step=step,
                source=source,
            )

        # 更新 operator_stats
        with _file_lock(self.operator_stats_path):
            op_stats = _load_json(self.operator_stats_path, {})

            for op_id in sorted(operators):
                if op_id not in op_stats:
                    new_operator_ids.append(op_id)
                    op_stats[op_id] = {
                        "operator_id": op_id,
                        "first_seen_step": step,
                        "first_seen_source": source,
                        "first_seen_time": ts,
                        "success_count": 0,
                        "failure_count": 0,
                        "placeholder_count": 0,
                        "parse_success_count": 0,
                        "parse_failure_count": 0,
                        "new_edge_count": 0,
                        "last_seen_step": None,
                        "last_seen_source": None,
                        "last_seen_time": None,
                    }

                op_stats[op_id]["success_count"] = int(op_stats[op_id].get("success_count", 0)) + 1
                op_stats[op_id]["parse_success_count"] = int(op_stats[op_id].get("parse_success_count", 0)) + int(parse_ok)
                op_stats[op_id]["last_seen_step"] = step
                op_stats[op_id]["last_seen_source"] = source
                op_stats[op_id]["last_seen_time"] = ts

            # 将新边贡献归因给 source operator。
            for edge in edges:
                e_key = edge_to_key(edge)
                if e_key in new_edge_keys:
                    op_id = edge.source_node_id
                    if op_id not in op_stats:
                        new_operator_ids.append(op_id)
                        op_stats[op_id] = {
                            "operator_id": op_id,
                            "first_seen_step": step,
                            "first_seen_source": source,
                            "first_seen_time": ts,
                            "success_count": 0,
                            "failure_count": 0,
                            "placeholder_count": 0,
                            "parse_success_count": 0,
                            "parse_failure_count": 0,
                            "new_edge_count": 0,
                            "last_seen_step": None,
                            "last_seen_source": None,
                            "last_seen_time": None,
                        }
                    op_stats[op_id]["new_edge_count"] = int(op_stats[op_id].get("new_edge_count", 0)) + 1

            _atomic_write_json(self.operator_stats_path, op_stats)

        # 更新 predicate_contribution
        with _file_lock(self.predicate_contribution_path):
            contrib = _load_json(self.predicate_contribution_path, {})

            if pred_key not in contrib:
                contrib[pred_key] = {
                    "predicate_key": pred_key,
                    "predicate": pred,
                    "normalized_predicate": normalized_predicate,
                    "abstract_predicate": abstract_predicate,
                    "first_seen_step": step,
                    "first_seen_source": source,
                    "first_seen_time": ts,
                    "success_count": 0,
                    "failure_count": 0,
                    "parse_success": False,
                    "parse_error": None,
                    "edges": [],
                    "operators": [],
                    "node_ids": [],
                    "new_edges": [],
                    "new_operators": [],
                    "graph_new_nodes": 0,
                    "graph_new_edges": 0,
                    "added_to_predicate": False,
                    "last_seen_step": None,
                    "last_seen_source": None,
                    "last_seen_time": None,
                }

            item = contrib[pred_key]
            item["success_count"] = int(item.get("success_count", 0)) + 1
            item["parse_success"] = bool(parse_ok)
            item["parse_error"] = parse_error
            item["normalized_predicate"] = normalized_predicate
            item["abstract_predicate"] = abstract_predicate
            item["edges"] = sorted(set(item.get("edges", []) + edge_keys))
            item["operators"] = sorted(set(item.get("operators", []) + list(operators)))
            item["node_ids"] = sorted(set(item.get("node_ids", []) + list(node_ids)))
            item["new_edges"] = sorted(set(item.get("new_edges", []) + new_edge_keys))
            item["new_operators"] = sorted(set(item.get("new_operators", []) + new_operator_ids))
            item["graph_new_nodes"] = int(item.get("graph_new_nodes", 0)) + int(graph_new_nodes)
            item["graph_new_edges"] = int(item.get("graph_new_edges", 0)) + int(graph_new_edges)
            item["last_seen_step"] = step
            item["last_seen_source"] = source
            item["last_seen_time"] = ts

            contrib[pred_key] = item
            _atomic_write_json(self.predicate_contribution_path, contrib)

        self._record_structure_attempt(
            predicate=pred,
            parse_result=parse_result,
            db_success=True,
            error=None,
            step=step,
            source=source,
            placeholder_count=0,
        )

        return {
            "predicate_key": pred_key,
            "parse_success": parse_ok,
            "edge_count": len(edge_keys),
            "new_edge_count": len(new_edge_keys),
            "operator_count": len(operators),
            "new_operator_count": len(new_operator_ids),
            "unknown_success": not parse_ok,
        }

    def record_sql_failure(
        self,
        predicate: str,
        parse_result: Optional[ParsedPredicateResult] = None,
        *,
        error: Optional[str] = None,
        step: Optional[int] = None,
        source: str = "unknown",
    ) -> Dict[str, Any]:
        """
        记录一次数据库执行失败的谓词。

        注意：
            失败不代表某条边不可达；
            这里只记录“这个具体谓词失败过”。
        """
        pred = "" if predicate is None else str(predicate)
        pred_key = _predicate_key(pred)
        operators = _parse_result_operators(parse_result)
        parse_ok = _parse_success(parse_result)

        with _file_lock(self.operator_stats_path):
            op_stats = _load_json(self.operator_stats_path, {})
            ts = _now_ts()

            for op_id in sorted(operators):
                if op_id not in op_stats:
                    op_stats[op_id] = {
                        "operator_id": op_id,
                        "first_seen_step": step,
                        "first_seen_source": source,
                        "first_seen_time": ts,
                        "success_count": 0,
                        "failure_count": 0,
                        "placeholder_count": 0,
                        "parse_success_count": 0,
                        "parse_failure_count": 0,
                        "new_edge_count": 0,
                        "last_seen_step": None,
                        "last_seen_source": None,
                        "last_seen_time": None,
                    }

                op_stats[op_id]["failure_count"] = int(op_stats[op_id].get("failure_count", 0)) + 1
                if parse_ok:
                    op_stats[op_id]["parse_success_count"] = int(op_stats[op_id].get("parse_success_count", 0)) + 1
                else:
                    op_stats[op_id]["parse_failure_count"] = int(op_stats[op_id].get("parse_failure_count", 0)) + 1
                op_stats[op_id]["last_seen_step"] = step
                op_stats[op_id]["last_seen_source"] = source
                op_stats[op_id]["last_seen_time"] = ts

            _atomic_write_json(self.operator_stats_path, op_stats)

        self._record_structure_attempt(
            predicate=pred,
            parse_result=parse_result,
            db_success=False,
            error=error,
            step=step,
            source=source,
            placeholder_count=0,
        )

        return {
            "predicate_key": pred_key,
            "parse_success": parse_ok,
            "operator_count": len(operators),
        }

    def record_joint_signature_type_failure(
        self,
        parse_result: Optional[ParsedPredicateResult],
        *,
        target_edge: Optional[Dict[str, Any]],
        target_edge_hit: bool,
        error_code: Optional[Any],
        error: Optional[str],
        timed_out: bool = False,
        sqlstate: Optional[str] = None,
        cancelled: bool = False,
    ) -> None:
        """Record targeted DBMS failure evidence without affecting Stage-2 decisions."""
        self.joint_signature_evidence.observe_target_type_failure(
            parse_result,
            target_edge=target_edge,
            target_edge_hit=target_edge_hit,
            error_code=error_code,
            error=error,
            timed_out=timed_out,
            **({"sqlstate": sqlstate, "cancelled": cancelled}
               if self.joint_signature_evidence.db_type == "sqlserver" else {}),
        )

    def flush_joint_signature_evidence(self) -> None:
        """Flush the bounded Stage-2 evidence buffer at a process boundary."""
        self.joint_signature_evidence.flush()

    def record_placeholder(
        self,
        predicate: str,
        placeholder_count: int,
        *,
        step: Optional[int] = None,
        source: str = "unknown",
    ) -> Dict[str, Any]:
        """
        记录一次含 ?? 的谓词。

        这种谓词不加入 Correct / Incorrect / Knowledge Graph，
        只作为坏样本统计。
        """
        pred = "" if predicate is None else str(predicate)
        pred_key = _predicate_key(pred)

        self._record_structure_attempt(
            predicate=pred,
            parse_result=None,
            db_success=False,
            error="placeholder detected",
            step=step,
            source=source,
            placeholder_count=int(placeholder_count),
        )

        return {
            "predicate_key": pred_key,
            "placeholder_count": int(placeholder_count),
        }

    def record_unknown_success(
        self,
        predicate: str,
        *,
        parse_error: Optional[str] = None,
        abstract_predicate: str = "",
        step: Optional[int] = None,
        source: str = "unknown",
    ) -> None:
        """
        记录 SQL 成功但 parser 失败的谓词。

        写入 unknown_success_buffer.txt，采用 JSON Lines 格式，每行一个 JSON。
        """
        payload = {
            "time": _now_ts(),
            "step": step,
            "source": source,
            "predicate": predicate,
            "abstract_predicate": abstract_predicate,
            "parse_error": parse_error,
        }
        _append_line_locked(
            self.unknown_success_buffer_path,
            json.dumps(payload, ensure_ascii=False, default=_json_default),
        )

    # ------------------------------------------------------------------
    # 内部结构尝试记录
    # ------------------------------------------------------------------

    def _record_structure_attempt(
        self,
        *,
        predicate: str,
        parse_result: Optional[ParsedPredicateResult],
        db_success: bool,
        error: Optional[str],
        step: Optional[int],
        source: str,
        placeholder_count: int,
    ) -> None:
        """
        更新 structure_attempts.json。

        这里按 predicate hash 聚合，不按理论边聚合。
        """
        pred = "" if predicate is None else str(predicate)
        pred_key = _predicate_key(pred)
        ts = _now_ts()

        parse_ok = _parse_success(parse_result)
        edges = _parse_result_edges(parse_result)
        operators = _parse_result_operators(parse_result)

        edge_keys = [edge_to_key(edge) for edge in edges]

        normalized_predicate = getattr(parse_result, "normalized_predicate", pred) if parse_result else pred
        abstract_predicate = getattr(parse_result, "abstract_predicate", "") if parse_result else ""
        parse_error = getattr(parse_result, "error", None) if parse_result else None

        with _file_lock(self.structure_attempts_path):
            attempts = _load_json(self.structure_attempts_path, {})

            if pred_key not in attempts:
                attempts[pred_key] = {
                    "predicate_key": pred_key,
                    "predicate": pred,
                    "normalized_predicate": normalized_predicate,
                    "abstract_predicate": abstract_predicate,
                    "first_seen_step": step,
                    "first_seen_source": source,
                    "first_seen_time": ts,
                    "attempt_count": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "placeholder_count": 0,
                    "parse_success_count": 0,
                    "parse_failure_count": 0,
                    "edges": [],
                    "operators": [],
                    "last_error": None,
                    "last_seen_step": None,
                    "last_seen_source": None,
                    "last_seen_time": None,
                }

            item = attempts[pred_key]
            item["attempt_count"] = int(item.get("attempt_count", 0)) + 1
            item["success_count"] = int(item.get("success_count", 0)) + int(bool(db_success))
            item["failure_count"] = int(item.get("failure_count", 0)) + int(not db_success)
            item["placeholder_count"] = int(item.get("placeholder_count", 0)) + int(placeholder_count)

            if parse_ok:
                item["parse_success_count"] = int(item.get("parse_success_count", 0)) + 1
            else:
                item["parse_failure_count"] = int(item.get("parse_failure_count", 0)) + 1

            item["normalized_predicate"] = normalized_predicate
            item["abstract_predicate"] = abstract_predicate
            item["edges"] = sorted(set(item.get("edges", []) + edge_keys))
            item["operators"] = sorted(set(item.get("operators", []) + list(operators)))
            item["last_error"] = error or parse_error
            item["last_seen_step"] = step
            item["last_seen_source"] = source
            item["last_seen_time"] = ts

            attempts[pred_key] = item
            _atomic_write_json(self.structure_attempts_path, attempts)


# ---------------------------------------------------------------------------
# 便捷单例接口
# ---------------------------------------------------------------------------

_DEFAULT_STATS: Optional[GraphRuntimeStats] = None


def get_default_runtime_stats() -> GraphRuntimeStats:
    """
    获取默认运行统计器。

    后续 env_sql_dependency.py / pkg_manager.py 可以直接调用这个函数。
    """
    global _DEFAULT_STATS
    if _DEFAULT_STATS is None:
        _DEFAULT_STATS = GraphRuntimeStats()
    return _DEFAULT_STATS


# ---------------------------------------------------------------------------
# 简单自测
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from tempfile import TemporaryDirectory
    from RL.predicate_structure import PredicateStructureAnalyzer

    with TemporaryDirectory() as tmp_dir:
        stats = GraphRuntimeStats(stats_dir=Path(tmp_dir))
        analyzer = PredicateStructureAnalyzer()

        sample = "(col(age) + int(1)) > int(10)"
        result = analyzer.analyze(sample)

        print("Parse success:", result.success)
        print("Edges:", [edge_to_key(edge) for edge in result.edges])

        summary = stats.record_sql_success(
            predicate=sample,
            parse_result=result,
            step=1,
            source="selftest",
            graph_new_nodes=0,
            graph_new_edges=len(result.edges),
        )

        print("Summary:", summary)
        print("Observed edges exists:", stats.observed_edges_path.exists())
        print("Structure attempts exists:", stats.structure_attempts_path.exists())
        print("Operator stats exists:", stats.operator_stats_path.exists())
        print("Predicate contribution exists:", stats.predicate_contribution_path.exists())
