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


MAX_DEPTH = 3


class DagUnfolder:
    def __init__(self, max_depth=3, deadline=None):
        self.max_depth = max_depth

        self.deadline = deadline

    @staticmethod
    def _copy_node_for_dag(node):
        copied = copy.copy(node)
        copied["slots"] = [copy.copy(slot) for slot in (node.get("slots", []) or [])]
        return copied

    def unfold(self, cyclic_graph_data):
        check_deadline(self.deadline, "DAG unfolding initialization")
        original_nodes = cyclic_graph_data['nodes']
        original_roots = cyclic_graph_data['roots']

        is_terminal_by_id = {
            node_id: bool(node['is_terminal'])
            for node_id, node in original_nodes.items()
        }

        dag_nodes = {}

        dag_roots = []

        for node_id, node in original_nodes.items():
            check_deadline(self.deadline, "DAG terminal expansion")
            if node['is_terminal']:
                dag_nodes[node_id] = self._copy_node_for_dag(node)

        for depth in range(self.max_depth + 1):
            check_deadline(self.deadline, f"DAG level {depth} expansion")
            for node_id, node in original_nodes.items():
                check_deadline(self.deadline, f"DAG level {depth} node expansion")

                if node['is_terminal']:
                    continue

                new_id = self._make_level_id(node_id, depth)

                new_node = self._copy_node_for_dag(node)
                new_node['original_id'] = node_id  

                new_slots = []
                for slot in node['slots']:
                    new_slot_successors = []
                    for succ_id in slot:
                        succ_is_terminal = is_terminal_by_id.get(succ_id)
                        if succ_is_terminal is None:
                            continue  

                        if succ_is_terminal:
                            new_slot_successors.append(succ_id)
                        else:
                            if depth < self.max_depth:
                                succ_level_id = self._make_level_id(succ_id, depth + 1)
                                new_slot_successors.append(succ_level_id)
                            else:
                                pass

                    new_slots.append(new_slot_successors)

                new_node['slots'] = new_slots
                dag_nodes[new_id] = new_node

                if depth == 0 and node_id in original_roots:
                    dag_roots.append(new_id)

        dag_nodes = self._prune_dead_nodes(dag_nodes)

        dag_roots = [r for r in dag_roots if r in dag_nodes]

        return {
            "roots": dag_roots,
            "nodes": dag_nodes
        }

    def _make_level_id(self, original_id, depth):
        return f"{original_id}::L{depth}"

    def _prune_dead_nodes(self, nodes):
        while True:
            check_deadline(self.deadline, "DAG dead-node pruning")
            nodes_to_remove = set()

            for nid, node in nodes.items():
                check_deadline(self.deadline, "DAG dead-node scan")
                if node['is_terminal']:
                    continue

                for slot in node['slots']:
                    if len(slot) == 0:
                        nodes_to_remove.add(nid)
                        break

            if not nodes_to_remove:
                break

            for nid in nodes_to_remove:
                del nodes[nid]

            for nid, node in nodes.items():
                if node['is_terminal']:
                    continue

                new_slots = []
                for slot in node['slots']:
                    cleaned_slot = [s_id for s_id in slot if s_id in nodes]
                    new_slots.append(cleaned_slot)
                node['slots'] = new_slots

        return nodes


def unfolder(max_depth=None, deadline=None):
    phase_timings = {
        "type_projection_seconds": 0.0,
        "dag_expansion_seconds": 0.0,
        "dag_output_seconds": 0.0,
    }

    base_dir = os.path.dirname(os.path.abspath(__file__))

    input_dir = os.path.join(base_dir, 'knowledge_graph')

    output_dir = os.path.join(base_dir, 'unfolded_graph')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

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

    selected_max_depth = MAX_DEPTH if max_depth is None else int(max_depth)
    if selected_max_depth < 0:
        raise ValueError("max_depth must be greater than or equal to 0")
    unfolder = DagUnfolder(max_depth=selected_max_depth, deadline=deadline)

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
        output_path = os.path.join(output_dir, f"dag_{file_name}")  

        try:
            check_deadline(deadline, "DAG file loading")
            print(f"Processing {file_name}...")

            with open(input_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)

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

            expansion_started = time.monotonic()
            try:
                dag_data = unfolder.unfold(graph_data)
            finally:
                phase_timings["dag_expansion_seconds"] += (
                    time.monotonic() - expansion_started
                )

            node_count = len(dag_data['nodes'])
            root_count = len(dag_data['roots'])

            if root_count == 0:
                print(f"  Warning: {file_name} resulted in an empty DAG (no valid roots after pruning).")

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
