import os
import json
import copy
import time
from pathlib import Path

from pkg.stage3_runtime import (
    Stage3DeadlineReached,
    atomic_json_dump,
    check_deadline,
)
from pkg.type_projection import project_graph_for_stage3

# --- 配置部分 ---
# 最大嵌套深度。论文建议限制嵌套深度以防止无限递归。
# L0 (Root) -> L1 -> L2 (Max) -> Terminals
# 设置为 3 表示允许最多嵌套 3 层算子 (例如: A AND (B OR (C > D)))
MAX_DEPTH = 3


class DagUnfolder:
    """
    负责将包含环的谓词知识图谱 (Cyclic Graph) 展开为有向无环图 (DAG)。
    核心逻辑：
    1. 按层级 (Level) 复制节点。
    2. 只有上一层的节点可以指向下一层的节点。
    3. 达到最大深度时，不再连接非终端节点。
    4. 清理掉那些参数槽位为空（无法实例化）的死节点。
    """

    def __init__(self, max_depth=3, deadline=None):
        # 初始化最大深度
        self.max_depth = max_depth
        # 限时固定深度实验可显式传入；None 保持默认深度。
        self.deadline = deadline

    @staticmethod
    def _copy_node_for_dag(node):
        """Copy only the structures that DAG expansion mutates.

        Expansion changes the top-level ``original_id`` and ``slots`` only.
        Copying every nested value with ``deepcopy`` in each depth is costly
        and does not change the resulting graph.
        """
        copied = copy.copy(node)
        copied["slots"] = [copy.copy(slot) for slot in (node.get("slots", []) or [])]
        return copied

    def unfold(self, cyclic_graph_data):
        """
        主入口：执行展开逻辑。
        :param cyclic_graph_data: 原始的有环图数据 (dict)
        :return: 展开后的 DAG 数据 (dict)
        """
        check_deadline(self.deadline, "DAG unfolding initialization")
        original_nodes = cyclic_graph_data['nodes']
        original_roots = cyclic_graph_data['roots']
        # The successor kind is stable throughout expansion.  A compact
        # lookup avoids repeating ``nodes.get`` plus a second dictionary access
        # for every edge at every depth.
        is_terminal_by_id = {
            node_id: bool(node['is_terminal'])
            for node_id, node in original_nodes.items()
        }

        # 用于存储展开后的新节点字典
        dag_nodes = {}
        # 用于存储展开后的新 Root ID
        dag_roots = []

        # --- 第一步：处理终端节点 (Terminals) ---
        # 终端节点 (如 col(), HOLE, 1, 'abc') 不需要分层，它们是共享的叶子。
        # 我们直接把它们复制进新图，保持原 ID 不变。
        for node_id, node in original_nodes.items():
            check_deadline(self.deadline, "DAG terminal expansion")
            if node['is_terminal']:
                # 深拷贝，防止修改原数据
                dag_nodes[node_id] = self._copy_node_for_dag(node)

        # --- 第二步：按深度展开算子节点 (Operators) ---
        # 我们从 depth = 0 遍历到 depth = max_depth - 1
        # 注意：实际上我们需要生成到 max_depth，但在 max_depth 层，算子只能接 Terminal

        for depth in range(self.max_depth + 1):
            check_deadline(self.deadline, f"DAG level {depth} expansion")
            for node_id, node in original_nodes.items():
                check_deadline(self.deadline, f"DAG level {depth} node expansion")
                # 只处理非终端节点 (算子)
                if node['is_terminal']:
                    continue

                # 生成带层级的新 ID，例如: "AND::T_BOOL" -> "AND::T_BOOL::L0"
                new_id = self._make_level_id(node_id, depth)

                # 复制节点基本信息
                new_node = self._copy_node_for_dag(node)
                new_node['original_id'] = node_id  # 记录“前世”ID，方便调试

                # --- 关键：重写 Slots (后继连接) ---
                # 我们需要修改 slots 里的每一个 successor ID
                new_slots = []
                for slot in node['slots']:
                    new_slot_successors = []
                    for succ_id in slot:
                        succ_is_terminal = is_terminal_by_id.get(succ_id)
                        if succ_is_terminal is None:
                            continue  # 异常保护

                        if succ_is_terminal:
                            # 情况 A: 后继是终端节点 (如 col, hole)
                            # 直接连接原始 ID (终端节点不分层)
                            new_slot_successors.append(succ_id)
                        else:
                            # 情况 B: 后继是算子 (如 AND, >)
                            # 只有当前深度 < 最大深度时，才允许连接下一层的算子
                            if depth < self.max_depth:
                                succ_level_id = self._make_level_id(succ_id, depth + 1)
                                new_slot_successors.append(succ_level_id)
                            else:
                                # 如果已经到了最大深度，就不允许再接算子了，直接丢弃这条边
                                # 这就是“剪枝”过程，防止无限递归
                                pass

                    new_slots.append(new_slot_successors)

                new_node['slots'] = new_slots
                dag_nodes[new_id] = new_node

                # 如果当前是第 0 层，且原节点是 Root，则将其加入 DAG Roots
                if depth == 0 and node_id in original_roots:
                    dag_roots.append(new_id)

        # --- 第三步：清理无效节点 (Pruning) ---
        # 论文提到: "subtrees with missing parameters cannot form a complete predicate"
        # 展开后，有些节点可能因为达到了最大深度，导致它的某个参数槽位变成了空的 (没有任何后继)。
        # 这种节点是废的，必须删掉。删掉它后，指向它的父节点可能也会变废，所以要循环清理。
        dag_nodes = self._prune_dead_nodes(dag_nodes)

        # --- 第四步：清理无效 Roots ---
        # 如果某个 Root 节点被清理掉了，也要从 roots 列表中移除
        dag_roots = [r for r in dag_roots if r in dag_nodes]

        return {
            "roots": dag_roots,
            "nodes": dag_nodes
        }

    def _make_level_id(self, original_id, depth):
        """生成带层级的 ID，格式: OriginalID::L{depth}"""
        return f"{original_id}::L{depth}"

    def _prune_dead_nodes(self, nodes):
        """
        循环清理死节点。
        死节点定义：非终端节点，且存在至少一个空的 Slot (参数缺失)。
        """
        while True:
            check_deadline(self.deadline, "DAG dead-node pruning")
            nodes_to_remove = set()

            # 1. 扫描所有节点，找出死节点
            for nid, node in nodes.items():
                check_deadline(self.deadline, "DAG dead-node scan")
                if node['is_terminal']:
                    continue

                # 检查每个槽位
                # 如果某个槽位 (list) 是空的，说明这个算子缺参数，无法构成完整谓词
                for slot in node['slots']:
                    if len(slot) == 0:
                        nodes_to_remove.add(nid)
                        break

            # 如果没有节点需要删除，说明图已经稳定，退出循环
            if not nodes_to_remove:
                break

            print(f"  [Pruning] Removing {len(nodes_to_remove)} dead nodes (incomplete parameters)...")

            # 2. 从字典中删除死节点
            for nid in nodes_to_remove:
                del nodes[nid]

            # 3. 更新剩余节点的 slots，移除指向死节点的边
            for nid, node in nodes.items():
                if node['is_terminal']:
                    continue

                # 遍历所有槽位，过滤掉已经删除的 successor ID
                new_slots = []
                for slot in node['slots']:
                    # 只保留那些还在 nodes 字典里的 ID
                    cleaned_slot = [s_id for s_id in slot if s_id in nodes]
                    new_slots.append(cleaned_slot)
                node['slots'] = new_slots

        return nodes


# --- 主执行逻辑 ---

def unfolder(max_depth=None, deadline=None):
    """
    展开知识图并写入 DAG 文件。

    max_depth 为 None 时保持原有固定深度 MAX_DEPTH=3；第三阶段限时实验会显式传入
    当前轮深度。这里只改变展开深度的来源，不改变 DAG 展开和剪枝算法。
    """
    phase_timings = {
        "type_projection_seconds": 0.0,
        "dag_expansion_seconds": 0.0,
        "dag_output_seconds": 0.0,
    }

    # 1. 定义路径
    # 获取当前脚本所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 输入目录: 之前生成的 knowledge_graph
    input_dir = os.path.join(base_dir, 'knowledge_graph')

    # 输出目录: 本次生成的 unfolded_graph
    output_dir = os.path.join(base_dir, 'unfolded_graph')

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 2. 检查输入目录
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return {
            "success": False,
            "round_complete": False,
            "phase": "unfolding",
            "error": f"Input directory not found: {input_dir}",
            "output_files": [],
            "phase_timings": phase_timings,
        }

    # 3. 初始化展开器。默认值保留原有行为，显式参数仅供第三阶段逐轮加深使用。
    selected_max_depth = MAX_DEPTH if max_depth is None else int(max_depth)
    if selected_max_depth < 0:
        raise ValueError("max_depth must be greater than or equal to 0")
    unfolder = DagUnfolder(max_depth=selected_max_depth, deadline=deadline)

    # 4. 遍历处理每个 JSON 文件
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

    if not json_files:
        print("No JSON files found in knowledge_graph folder.")
        return {
            "success": False,
            "round_complete": False,
            "phase": "unfolding",
            "error": "No JSON files found in knowledge_graph folder.",
            "output_files": [],
            "phase_timings": phase_timings,
        }

    print(
        f"Found {len(json_files)} graph files. "
        f"Starting unfolding (Max Depth = {selected_max_depth})..."
    )

    output_files = []
    projection_reports = []
    for file_name in sorted(json_files):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, f"dag_{file_name}")  # 加上前缀区分

        try:
            check_deadline(deadline, "DAG file loading")
            print(f"Processing {file_name}...")

            # 读取
            with open(input_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)

            # Stage-3 only: split the in-memory graph by complete DBMS type
            # signatures. The Stage-2 global graph file is never rewritten.
            projection_started = time.monotonic()
            try:
                graph_data, projection_report = project_graph_for_stage3(
                    graph_data,
                    source_path=Path(input_path),
                    deadline=deadline,
                )
            finally:
                phase_timings["type_projection_seconds"] += (
                    time.monotonic() - projection_started
                )
            projection_reports.append(projection_report)
            print(
                "  [TypeProjection] "
                f"status={projection_report.get('status')}, "
                f"nodes={projection_report.get('original_nodes')}->"
                f"{projection_report.get('projected_nodes')}, "
                f"edges={projection_report.get('original_edges')}->"
                f"{projection_report.get('projected_edges')}"
            )

            # 展开
            expansion_started = time.monotonic()
            try:
                dag_data = unfolder.unfold(graph_data)
            finally:
                phase_timings["dag_expansion_seconds"] += (
                    time.monotonic() - expansion_started
                )

            # 检查结果
            node_count = len(dag_data['nodes'])
            root_count = len(dag_data['roots'])

            if root_count == 0:
                print(f"  Warning: {file_name} resulted in an empty DAG (no valid roots after pruning).")
            else:
                print(f"  Success! Generated DAG with {node_count} nodes and {root_count} roots.")

            # 先写 .partial，成功关闭后再替换正式 DAG，防止异常留下半截文件。
            output_started = time.monotonic()
            try:
                atomic_json_dump(
                    dag_data,
                    output_path,
                    deadline=deadline,
                    phase="DAG JSON output",
                )
            finally:
                phase_timings["dag_output_seconds"] += (
                    time.monotonic() - output_started
                )
            output_files.append(output_path)

        except Stage3DeadlineReached as e:
            print(f"  [Stage3] Deadline reached: {e.phase}")
            return {
                "success": True,
                "round_complete": False,
                "phase": "unfolding",
                "error": None,
                "output_files": output_files,
                "projection_reports": projection_reports,
                "phase_timings": phase_timings,
            }
        except Exception as e:
            print(f"  Error processing {file_name}: {e}")
            return {
                "success": False,
                "round_complete": False,
                "phase": "unfolding",
                "error": str(e),
                "output_files": output_files,
                "projection_reports": projection_reports,
                "phase_timings": phase_timings,
            }

    print("All unfolding tasks completed.")
    return {
        "success": True,
        "round_complete": True,
        "phase": "unfolding",
        "error": None,
        "output_files": output_files,
        "projection_reports": projection_reports,
        "phase_timings": phase_timings,
    }


if __name__ == "__main__":
    unfolder()
