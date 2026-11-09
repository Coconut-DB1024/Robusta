import os  
import json  

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
    return format_large_int(value, max_bits=max_bits)


class SubtreeCounter:
    def __init__(self):
        self.last_count_stats = {
            "count_nodes": 0,
            "count_bits": 0,
            "estimated_bytes": 0,
        }

    def count(self, dag_data, deadline=None, memory_budget_bytes=None):
        check_deadline(deadline, "counting initialization")
        nodes = dag_data['nodes']
        count_nodes = 0
        count_bits = 0

        def record_count(value, node_id):
            nonlocal count_nodes, count_bits
            count_nodes += 1
            count_bits += max(1, int(value).bit_length())

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

        terminals = []

        operators_by_level = {}

        for nid, node in nodes.items():
            check_deadline(deadline, "counting node classification")

            if node['is_terminal']:
                terminals.append(nid)  
            else:
                try:
                    level_str = nid.split('::')[-1]
                    level = int(level_str.replace('L', ''))

                    if level not in operators_by_level:
                        operators_by_level[level] = []

                    operators_by_level[level].append(nid)
                except ValueError:
                    print(f"Warning: Node {nid} has invalid level format, skipping.")

        for nid in terminals:
            check_deadline(deadline, "terminal count initialization")
            nodes[nid]['count'] = 1
            record_count(1, nid)

        levels = sorted(operators_by_level.keys(), reverse=True)

        for level in levels:
            check_deadline(deadline, f"counting level {level}")

            current_level_nodes = operators_by_level[level]

            for nid in current_level_nodes:
                check_deadline(deadline, f"counting node {nid}")
                node = nodes[nid]  

                total_combinations = 1

                for slot in node['slots']:
                    check_deadline(deadline, f"counting slots of {nid}")

                    slot_sum = 0

                    for child_id in slot:
                        check_deadline(deadline, f"counting successors of {nid}")

                        if child_id in nodes:
                            child_count = nodes[child_id].get('count', 0)

                            slot_sum += child_count
                        else:
                            print(f"Warning: Node {nid} points to missing child {child_id}")

                    total_combinations *= slot_sum

                node['count'] = total_combinations
                record_count(total_combinations, nid)

        return dag_data


def counter(deadline=None, input_files=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    input_dir = os.path.join(base_dir, 'unfolded_graph')

    output_dir = os.path.join(base_dir, 'counted_graph')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  
        print(f"Created output directory: {output_dir}")  

    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return {
            "success": False,
            "round_complete": False,
            "phase": "counting",
            "error": f"Input directory not found: {input_dir}",
            "output_files": [],
        }

    subtree_counter = SubtreeCounter()
    budget_mb = int(
        getattr(STAGE3_CONFIG, "stage3_count_memory_budget_mb", 256)
        if STAGE3_CONFIG is not None
        else 256
    )
    memory_budget_bytes = budget_mb * 1024 * 1024 if budget_mb > 0 else None

    if input_files is None:
        json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    else:
        json_files = [
            os.path.basename(str(path))
            for path in input_files
            if str(path).endswith('.json')
        ]

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

    output_files = []
    for file_name in sorted(json_files):
        output_name = "counted_" + file_name

        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, output_name)

        try:
            check_deadline(deadline, "counted DAG input loading")

            with open(input_path, 'r', encoding='utf-8') as f:
                dag_data = json.load(f)

            counted_data = subtree_counter.count(
                dag_data,
                deadline=deadline,
                memory_budget_bytes=memory_budget_bytes,
            )

            roots = counted_data['roots']
            total_possibilities = 0
            for r in roots:
                if r in counted_data['nodes']:
                    total_possibilities += counted_data['nodes'][r]['count']

            encode_counts_in_place(counted_data, deadline=deadline)

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

    print("Counting tasks completed.")  
    return {
        "success": True,
        "round_complete": True,
        "phase": "counting",
        "error": None,
        "output_files": output_files,
        "resource_limited": False,
        "count_stats": subtree_counter.last_count_stats,
    }


if __name__ == "__main__":
    counter()
