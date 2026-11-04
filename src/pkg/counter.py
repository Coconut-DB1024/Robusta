import os  # 导入操作系统模块，用于处理文件路径和目录创建
import json  # 导入json模块，用于读取和保存图谱数据

from pkg.stage3_runtime import (
    Stage3DeadlineReached,
    Stage3ResourceLimitReached,
    atomic_json_dump,
    check_deadline,
    encode_counts_in_place,
    format_large_int,
)

try:
    from RL.config import STAGE3_CONFIG
except Exception:
    STAGE3_CONFIG = None


def _format_count_for_log(value, max_bits=256):
    """限制超大组合数的日志表示，避免计数正确但控制台输出失控。"""
    return format_large_int(value, max_bits=max_bits)


class SubtreeCounter:
    """
    子树计数器类。
    功能：实现论文中的公式 ψ(t) = Π (Σ ψ(child))。
    逻辑：采用动态规划思想，自底向上（从叶子到根）计算每个节点能生成的谓词组合总数。
    """

    def __init__(self):
        self.last_count_stats = {
            "count_nodes": 0,
            "count_bits": 0,
            "estimated_bytes": 0,
        }

    def count(self, dag_data, deadline=None, memory_budget_bytes=None):
        """
        核心计算函数。
        输入：dag_data (dict) - 包含 'nodes' 和 'roots' 的无环图数据。
        输出：dag_data (dict) - 节点信息中增加了 'count' 属性的图数据。
        """
        # 获取所有节点的字典引用
        check_deadline(deadline, "counting initialization")
        nodes = dag_data['nodes']
        count_nodes = 0
        count_bits = 0

        def record_count(value, node_id):
            nonlocal count_nodes, count_bits
            count_nodes += 1
            count_bits += max(1, int(value).bit_length())
            # Python int 除原始 limb 数据外还有对象头；每个计数按 32 字节估算固定开销。
            estimated_bytes = (count_bits + 7) // 8 + count_nodes * 32
            self.last_count_stats = {
                "count_nodes": count_nodes,
                "count_bits": count_bits,
                "estimated_bytes": estimated_bytes,
            }
            if (
                memory_budget_bytes is not None
                and memory_budget_bytes > 0
                and estimated_bytes > memory_budget_bytes
            ):
                raise Stage3ResourceLimitReached(
                    "exact combination counting",
                    (
                        f"estimated_count_memory={estimated_bytes} bytes exceeds "
                        f"budget={memory_budget_bytes} bytes at node {node_id}"
                    ),
                )

        # --- 第一步：节点分类 (分层) ---
        # 我们需要按照依赖关系顺序计算：先算叶子，再算底层算子，最后算顶层算子。

        # 用于存储终端节点（叶子）的ID列表
        terminals = []
        # 用于存储算子节点的字典，Key为层级(int)，Value为该层级的节点ID列表
        # 例如: {3: ['AND::L3', ...], 2: ['OR::L2', ...]}
        operators_by_level = {}

        # 遍历图中的每一个节点进行分类
        for nid, node in nodes.items():
            check_deadline(deadline, "counting node classification")
            # 判断是否为终端节点 (is_terminal 为 true 的节点，如列名、数字、HOLE)
            if node['is_terminal']:
                terminals.append(nid)  # 加入终端列表
            else:
                # 如果是算子节点，解析其层级 (Level)
                # 节点ID格式通常为: "TOKEN::TYPE::L{n}"，例如 "AND::T_BOOL::L2"
                try:
                    # 取ID的最后一部分，去掉 'L'，转为整数
                    level_str = nid.split('::')[-1]
                    level = int(level_str.replace('L', ''))

                    # 如果该层级还没有记录，初始化一个空列表
                    if level not in operators_by_level:
                        operators_by_level[level] = []
                    # 将节点ID加入对应层级的列表
                    operators_by_level[level].append(nid)
                except ValueError:
                    # 如果ID格式不对（理论上不应发生），打印警告并跳过
                    print(f"Warning: Node {nid} has invalid level format, skipping.")

        # --- 第二步：初始化终端节点计数 ---
        # 根据定义，终端节点本身就是一颗完整的微型子树，没有变化，所以计数为 1。
        for nid in terminals:
            check_deadline(deadline, "terminal count initialization")
            nodes[nid]['count'] = 1
            record_count(1, nid)

        # --- 第三步：自底向上动态规划计算 ---
        # 获取所有存在的层级，并按降序排列 (例如: [3, 2, 1, 0])
        # 必须先算深层 (Level 3)，因为浅层 (Level 2) 依赖深层的结果。
        levels = sorted(operators_by_level.keys(), reverse=True)

        # 逐层遍历
        for level in levels:
            check_deadline(deadline, f"counting level {level}")
            # 获取当前层级的所有节点ID
            current_level_nodes = operators_by_level[level]

            # 遍历当前层的每一个节点
            for nid in current_level_nodes:
                check_deadline(deadline, f"counting node {nid}")
                node = nodes[nid]  # 获取节点对象

                # 初始化当前节点的组合总数为 1 (乘法的单位元)
                # 公式：ψ(t) = Product( Slot_Sum )
                total_combinations = 1

                # 遍历该节点的每一个插槽 (Slot)
                # Slot 代表算子的参数位置，例如 A + B 有两个 Slot (左操作数, 右操作数)
                for slot in node['slots']:
                    check_deadline(deadline, f"counting slots of {nid}")
                    # 初始化当前插槽的变化数为 0 (加法的单位元)
                    # 公式内部：Sum( child_count )
                    slot_sum = 0

                    # 遍历该插槽内所有可能的子节点 (Successor)
                    for child_id in slot:
                        check_deadline(deadline, f"counting successors of {nid}")
                        # 检查子节点是否存在于图谱中 (安全检查)
                        if child_id in nodes:
                            # 获取子节点的已计算出的 count 值
                            # dict.get(key, 0) 防止意外缺失，默认为0
                            child_count = nodes[child_id].get('count', 0)
                            # 累加到插槽总数中
                            slot_sum += child_count
                        else:
                            # 如果指向了不存在的节点，打印警告
                            print(f"Warning: Node {nid} points to missing child {child_id}")

                    # 将当前插槽的总变化数乘到节点总数中
                    # 原理：不同参数槽位之间的组合是笛卡尔积关系 (乘法原理)
                    total_combinations *= slot_sum

                # 将计算结果保存到节点属性中
                node['count'] = total_combinations
                record_count(total_combinations, nid)

        # --- 第四步：返回结果 ---
        # 此时 dag_data 中的 nodes 已经被修改，包含了 count 属性
        return dag_data


# --- 主执行逻辑 ---

def counter(deadline=None, input_files=None):
    # 1. 获取当前脚本所在的绝对路径
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. 定义输入目录：读取 'unfolded_graph' 文件夹 (上一步的产物)
    input_dir = os.path.join(base_dir, 'unfolded_graph')

    # 3. 定义输出目录：保存到 'counted_graph' 文件夹 (本次新建)
    output_dir = os.path.join(base_dir, 'counted_graph')

    # 4. 检查输出目录是否存在，不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 创建目录
        print(f"Created output directory: {output_dir}")  # 打印提示

    # 5. 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return {
            "success": False,
            "round_complete": False,
            "phase": "counting",
            "error": f"Input directory not found: {input_dir}",
            "output_files": [],
        }

    # 6. 初始化计数器实例
    subtree_counter = SubtreeCounter()
    budget_mb = int(
        getattr(STAGE3_CONFIG, "stage3_count_memory_budget_mb", 256)
        if STAGE3_CONFIG is not None
        else 256
    )
    memory_budget_bytes = budget_mb * 1024 * 1024 if budget_mb > 0 else None

    # 7. 获取输入目录下的所有 JSON 文件列表
    if input_files is None:
        json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    else:
        json_files = [
            os.path.basename(str(path))
            for path in input_files
            if str(path).endswith('.json')
        ]

    # 如果没有找到文件，打印提示并退出
    if not json_files:
        print("No JSON files found in unfolded_graph folder.")
        return {
            "success": False,
            "round_complete": False,
            "phase": "counting",
            "error": "No JSON files found in unfolded_graph folder.",
            "output_files": [],
        }

    print(f"Found {len(json_files)} DAG files. Starting counting...")

    # 8. 循环处理每一个文件
    output_files = []
    for file_name in sorted(json_files):
        # 构造输出文件名：在原文件名前加上 "counted_" 前缀
        # 例如: dag_pkg_batch_1.json -> counted_dag_pkg_batch_1.json
        output_name = "counted_" + file_name

        # 构造完整的输入和输出文件路径
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, output_name)

        try:
            check_deadline(deadline, "counted DAG input loading")
            # 读取 JSON 文件
            with open(input_path, 'r', encoding='utf-8') as f:
                dag_data = json.load(f)

            # 调用核心计算逻辑
            counted_data = subtree_counter.count(
                dag_data,
                deadline=deadline,
                memory_budget_bytes=memory_budget_bytes,
            )

            # --- 简单的统计信息展示 (可选) ---
            # 计算一下所有 Root 节点的组合数之和，让人心里有底
            roots = counted_data['roots']
            total_possibilities = 0
            for r in roots:
                if r in counted_data['nodes']:
                    total_possibilities += counted_data['nodes'][r]['count']

            # 超大整数使用可逆十六进制字符串保存，绕过十进制 4300 位限制。
            # 数值本身仍然完全精确，采样阶段会恢复为 Python int。
            encode_counts_in_place(counted_data, deadline=deadline)

            # 原子写入避免异常或强制终止把正式 counted DAG 截成半个 JSON。
            atomic_json_dump(
                counted_data,
                output_path,
                deadline=deadline,
                phase="counted DAG JSON output",
            )
            output_files.append(output_path)

        except Stage3DeadlineReached as e:
            print(f"[Stage3] Deadline reached during counting: {e.phase}")
            return {
                "success": True,
                "round_complete": False,
                "phase": "counting",
                "error": None,
                "output_files": output_files,
                "count_stats": subtree_counter.last_count_stats,
            }
        except Stage3ResourceLimitReached as e:
            print(f"[Stage3] Count resource limit reached: {e.detail}")
            return {
                "success": True,
                "round_complete": False,
                "phase": "counting",
                "error": None,
                "resource_limited": True,
                "resource_detail": e.detail,
                "output_files": output_files,
                "count_stats": subtree_counter.last_count_stats,
            }
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            return {
                "success": False,
                "round_complete": False,
                "phase": "counting",
                "error": str(e),
                "output_files": output_files,
                "count_stats": subtree_counter.last_count_stats,
            }

    print("Counting tasks completed.")  # 任务完成提示
    return {
        "success": True,
        "round_complete": True,
        "phase": "counting",
        "error": None,
        "output_files": output_files,
        "resource_limited": False,
        "count_stats": subtree_counter.last_count_stats,
    }


# 程序入口保护
if __name__ == "__main__":
    counter()
