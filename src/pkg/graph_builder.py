import json


class GraphNode:
    def __init__(self, token, return_type, is_terminal=False):
        self.id = f"{token}::{return_type}"  
        self.token = token
        self.return_type = return_type
        self.is_terminal = is_terminal

        self.slots = []

    def ensure_slots(self, count):
        while len(self.slots) < count:
            self.slots.append(set())

    def add_successor(self, slot_index, child_node_id):
        self.ensure_slots(slot_index + 1)

        if child_node_id not in self.slots[slot_index]:
            self.slots[slot_index].add(child_node_id)
            return True  
        return False  

    def to_dict(self):
        return {
            "token": self.token,
            "return_type": self.return_type,
            "is_terminal": self.is_terminal,

            "slots": [list(s) for s in self.slots]
        }

    @classmethod
    def from_dict(cls, node_id, payload):
        node = cls(
            token=payload.get("token", ""),
            return_type=payload.get("return_type", ""),
            is_terminal=bool(payload.get("is_terminal", False)),
        )
        node.id = node_id
        node.slots = [set(slot_items) for slot_items in payload.get("slots", [])]
        return node


class KnowledgeGraphBuilder:
    def __init__(self):
        self.nodes = {}

        self.roots = set()

        self.new_nodes_count = 0  
        self.new_edges_count = 0  
        self.total_edges = 0  

    def get_or_create_node(self, ast_node):
        is_terminal = (ast_node.node_type != 'operator')

        node_id = f"{ast_node.token}::{ast_node.return_type}"

        if node_id not in self.nodes:
            new_node = GraphNode(ast_node.token, ast_node.return_type, is_terminal)
            self.nodes[node_id] = new_node
            self.new_nodes_count += 1  

        return self.nodes[node_id]

    def initialize_nodes_from_candidate_records(self, candidate_records):
        self.new_nodes_count = 0
        self.new_edges_count = 0

        node_specs = {}
        for record in candidate_records or []:
            if not isinstance(record, dict):
                continue

            for role in ("outer", "inner"):
                node_id = str(record.get(f"{role}_node_id", "") or "")
                if not node_id:
                    continue
                node_specs.setdefault(
                    node_id,
                    {
                        "token": str(record.get(f"{role}_token", "") or ""),
                        "return_type": str(record.get(f"{role}_return_type", "") or ""),
                        "slot_count": 0,
                    },
                )

            outer_node_id = str(record.get("outer_node_id", "") or "")
            if not outer_node_id:
                continue
            try:
                slot_index = int(record.get("slot_index", -1))
            except Exception:
                continue
            if slot_index >= 0 and outer_node_id in node_specs:
                node_specs[outer_node_id]["slot_count"] = max(
                    int(node_specs[outer_node_id]["slot_count"]),
                    slot_index + 1,
                )

        for node_id, spec in node_specs.items():
            node = self.nodes.get(node_id)
            if node is None:
                node = GraphNode(
                    spec["token"],
                    spec["return_type"],
                    is_terminal=int(spec["slot_count"]) == 0,
                )
                node.id = node_id
                self.nodes[node_id] = node
                self.new_nodes_count += 1
            node.ensure_slots(int(spec["slot_count"]))

        return self.new_nodes_count, self.new_edges_count

    def add_candidate_target_edge(
        self,
        *,
        outer_node_id,
        outer_token,
        outer_return_type,
        slot_index,
        inner_node_id,
        inner_token,
        inner_return_type,
    ):
        self.new_nodes_count = 0
        self.new_edges_count = 0

        try:
            slot_index = int(slot_index)
        except Exception as exc:
            raise ValueError(f"invalid candidate slot index: {slot_index}") from exc
        if slot_index < 0:
            raise ValueError(f"candidate slot index must be non-negative: {slot_index}")

        outer_node_id = str(outer_node_id)
        inner_node_id = str(inner_node_id)
        if not outer_node_id or not inner_node_id:
            raise ValueError("candidate target edge requires both node ids")

        outer_node = self.nodes.get(outer_node_id)
        if outer_node is None:
            outer_node = GraphNode(str(outer_token), str(outer_return_type), is_terminal=False)
            outer_node.id = outer_node_id
            self.nodes[outer_node_id] = outer_node
            self.new_nodes_count += 1

        inner_node = self.nodes.get(inner_node_id)
        if inner_node is None:
            inner_node = GraphNode(str(inner_token), str(inner_return_type), is_terminal=True)
            inner_node.id = inner_node_id
            self.nodes[inner_node_id] = inner_node
            self.new_nodes_count += 1

        if outer_node.add_successor(slot_index, inner_node.id):
            self.new_edges_count = 1
            self.total_edges += 1

        self.roots.add(outer_node.id)
        return self.new_nodes_count, self.new_edges_count

    def get_slot_index(self, ast_node, child_index):
        if ast_node.token.upper() in {"IN", "NOT IN"} and child_index >= 1:
            return 1
        return child_index

    def merge_ast(self, ast_root):
        if not ast_root:
            return

        current_graph_node = self.get_or_create_node(ast_root)

        pass

        for i, child_ast in enumerate(ast_root.children):
            child_graph_node = self.get_or_create_node(child_ast)

            slot_index = self.get_slot_index(ast_root, i)
            is_new_edge = current_graph_node.add_successor(slot_index, child_graph_node.id)
            if is_new_edge:
                self.new_edges_count += 1
                self.total_edges += 1

            self.merge_ast(child_ast)

        return current_graph_node.id

    def add_predicate(self, ast_root):
        if not ast_root:
            return 0, 0

        self.new_nodes_count = 0
        self.new_edges_count = 0

        root_id = self.merge_ast(ast_root)
        self.roots.add(root_id)

        return self.new_nodes_count, self.new_edges_count

    def save_to_json(self, filepath):
        output = {
            "roots": list(self.roots),
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()}
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    def load_from_json(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.nodes = {}
        for node_id, payload in data.get("nodes", {}).items():
            self.nodes[node_id] = GraphNode.from_dict(node_id, payload)

        self.roots = set(data.get("roots", []))

        self.new_nodes_count = 0
        self.new_edges_count = 0

        self.total_edges = 0
        for node in self.nodes.values():
            for slot_successors in node.slots:
                self.total_edges += len(slot_successors)

        return self

    def iter_edges(self):
        for outer_node_id, node in self.nodes.items():
            for slot_index, slot_successors in enumerate(node.slots):
                for inner_node_id in sorted(slot_successors):
                    yield outer_node_id, slot_index, inner_node_id

    def get_node_ids(self):
        return sorted(self.nodes.keys())

    def get_edge_count(self):
        count = 0
        for _, _, _ in self.iter_edges():
            count += 1
        return count
