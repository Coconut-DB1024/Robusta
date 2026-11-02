import json


class GraphNode:
    def __init__(self, token, return_type, is_terminal=False):
        self.id = f"{token}::{return_type}"  # 唯一标识符
        self.token = token
        self.return_type = return_type
        self.is_terminal = is_terminal
        # slots 是一个列表，列表每项是一个 set（存储该槽位所有可能的后继节点ID）
        self.slots = []

    def ensure_slots(self, count):
        """确保槽位数量足够"""
        while len(self.slots) < count:
            self.slots.append(set())

    def add_successor(self, slot_index, child_node_id):
        self.ensure_slots(slot_index + 1)
        # self.slots[slot_index].add(child_node_id)
        # 【修改】：判断子节点是否已经存在于该槽位中，如果不存在，说明这是一条“新发现的语义依赖边”
        if child_node_id not in self.slots[slot_index]:
            self.slots[slot_index].add(child_node_id)
            return True  # 返回 True 表示添加了新边
        return False  # 返回 False 表示边已存在

    def to_dict(self):
        """转为 JSON 友好的格式"""
        return {
            "token": self.token,
            "return_type": self.return_type,
            "is_terminal": self.is_terminal,
            # 将 set 转为 list 以便 JSON 序列化
            "slots": [list(s) for s in self.slots]
        }

    @classmethod
    def from_dict(cls, node_id, payload):
        """
        从 JSON 字典恢复 GraphNode。

        save_to_json() 保存的是 token / return_type / is_terminal / slots；
        slots 在 JSON 中是 list[list[str]]，恢复时需要转回 list[set[str]]。
        """
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
        # 节点注册表: {node_id: GraphNode}
        self.nodes = {}
        # 记录哪些节点是整个谓词的起点 (Roots)
        self.roots = set()
        # --- 【新增：增量计数器】 ---
        self.new_nodes_count = 0  # 单次插入产生的新节点数
        self.new_edges_count = 0  # 单次插入产生的新边数
        self.total_edges = 0  # 当前图谱的总边数
        # ---------------------------

    def get_or_create_node(self, ast_node):
        """根据 AST 节点获取图节点，如果不存在则创建"""
        # 区分终端节点 (HOLE, Column, Literal) 和 算子
        is_terminal = (ast_node.node_type != 'operator')

        # 唯一 ID 生成规则
        node_id = f"{ast_node.token}::{ast_node.return_type}"

        if node_id not in self.nodes:
            new_node = GraphNode(ast_node.token, ast_node.return_type, is_terminal)
            self.nodes[node_id] = new_node
            self.new_nodes_count += 1  # 【新增】：记录发现了新节点

        return self.nodes[node_id]

    def initialize_nodes_from_candidate_records(self, candidate_records):
        """Create the candidate-universe nodes and slots without adding relation edges."""
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
        """Insert exactly one candidate edge after its target probe succeeds."""
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
        """将一棵 Typed AST 融合进图谱"""
        if not ast_root:
            return

        # 1. 获取/创建当前节点
        current_graph_node = self.get_or_create_node(ast_root)

        # 2. 如果是根节点调用，记录到 roots
        # (这里外部调用时需要注意，递归内部调用时不加)
        pass

        # 3. 递归处理子节点并建立连接
        for i, child_ast in enumerate(ast_root.children):
            child_graph_node = self.get_or_create_node(child_ast)

            # 建立边：Current[Slot i] -> Child
            #current_graph_node.add_successor(i, child_graph_node.id)
            # 【修改】：接收 add_successor 的返回值，如果是新边则累加计数
            slot_index = self.get_slot_index(ast_root, i)
            is_new_edge = current_graph_node.add_successor(slot_index, child_graph_node.id)
            if is_new_edge:
                self.new_edges_count += 1
                self.total_edges += 1

            # 递归融合子树
            self.merge_ast(child_ast)

        return current_graph_node.id

    def add_predicate(self, ast_root):
        """添加一条完整的谓词 AST"""
        if not ast_root:
            return 0, 0

        # 【新增】：每次插入新谓词前，清空本次的增量计数
        self.new_nodes_count = 0
        self.new_edges_count = 0

        root_id = self.merge_ast(ast_root)
        self.roots.add(root_id)

        # 【新增】：返回本次插入发现的新节点和新边数量，供强化学习监控使用！
        return self.new_nodes_count, self.new_edges_count

    def save_to_json(self, filepath):
        """保存图谱"""
        output = {
            "roots": list(self.roots),
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()}
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    def load_from_json(self, filepath):
        """
        从 global_knowledge_graph.json 恢复图谱状态。
        """
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
        """
        阶段17-1 / 17-2：统一遍历知识图中的语义依赖边。

        yield:
            outer_node_id, slot_index, inner_node_id
        """
        for outer_node_id, node in self.nodes.items():
            for slot_index, slot_successors in enumerate(node.slots):
                for inner_node_id in sorted(slot_successors):
                    yield outer_node_id, slot_index, inner_node_id

    def get_node_ids(self):
        """
        返回当前知识图中的所有节点 ID。
        """
        return sorted(self.nodes.keys())

    def get_edge_count(self):
        """
        返回当前知识图中的边数量。
        """
        count = 0
        for _, _, _ in self.iter_edges():
            count += 1
        return count
