class TypeInferencer:
    def __init__(self, config):
        self.config = config or {}  # 【必须增加 or {}】：防御 NoneType 崩溃
        self.definitions = self.config.get("definitions", {})
        self.promotions = self.config.get("promotions", {})
        self.candidate_bitmap_config = self.config.get("candidate_bitmap", {}) or {}
        self.coarse_type_aliases = self.candidate_bitmap_config.get("coarse_type_aliases", {}) or {}
        self.callable_terminal_tokens = {
            str(token).strip().upper()
            for token in self.candidate_bitmap_config.get("callable_terminal_tokens", []) or []
            if str(token).strip()
        }
        self.configured_terminal_tokens = {
            str(token).strip().upper()
            for token in (self.candidate_bitmap_config.get("terminal_tokens", {}) or {}).keys()
            if str(token).strip()
        }

    def normalize_candidate_type(self, raw_type):
        """
        阶段17-1：将 INT / VARCHAR / BOOL 等具体类型归一化为候选边构建使用的粗粒度类型。

        例如：
            INT      -> T_CMP
            VARCHAR  -> T_STR
            BOOL     -> T_BOOL
            T_ANY    -> T_ANY
        """
        if raw_type is None:
            return "T_ANY"

        text = str(raw_type).strip().upper()
        if not text:
            return "T_ANY"

        return str(self.coarse_type_aliases.get(text, text)).upper()

    def get_operator_signature(self, token):
        """
        阶段17-1：读取某个算子的返回类型和参数槽位类型。

        返回：
            {
                "token": token,
                "return_type": "...",
                "arg_types": [...]
            }

        如果规则中不存在该 token，返回 None。
        """
        op_token = str(token).strip().upper()
        rule = self.definitions.get(op_token)
        if not isinstance(rule, list) or len(rule) < 2:
            return None

        return {
            "token": op_token,
            "return_type": self.normalize_candidate_type(rule[0]),
            "arg_types": [self.normalize_candidate_type(x) for x in rule[1]],
        }

    def get_terminal_return_type(self, token):
        """
        阶段17-1：推断 COL_* / VALUE_* / NULL / TRUE / FALSE 等终端 token 的粗粒度返回类型。
        """
        text = str(token).strip().upper()
        terminal_tokens = self.candidate_bitmap_config.get("terminal_tokens", {}) or {}

        if text in terminal_tokens:
            return self.normalize_candidate_type(terminal_tokens[text])

        if text.startswith("COL_"):
            return self.normalize_candidate_type(text.split("_", 1)[1])

        if text.startswith("VALUE_"):
            parts = text.split("_")
            if parts:
                return self.normalize_candidate_type(parts[-1])

        if text in {"TRUE", "FALSE"}:
            return "T_BOOL"

        if text == "NULL":
            return "T_ANY"

        return "T_ANY"

    def infer(self, node, expected_type="T_BOOL"):
        """
        递归地推断 AST 节点类型，并进行 '类型填坑'。
        node: ASTNode
        expected_type: 父节点期望该节点是什么类型
        """
        if not node:
            return None

        # --- 1. 处理 HOLE (??) ---
        if node.node_type == 'hole':
            # 核心逻辑：根据父节点的期望，赋予 HOLE 具体身份
            # 如果期望是 Any，默认给 T_CMP (MySQL中最通用)
            target_type = expected_type if expected_type != "T_ANY" else "T_CMP"

            # 修改节点 Token，使其在图谱中唯一
            # 例如 ?? -> HOLE_T_STR
            node.token = f"HOLE_{target_type}"
            node.return_type = target_type
            return target_type

        # --- 2. 处理 Column / Literal ---
        """
        if node.node_type == 'column' or node.node_type == 'literal':
            # 策略：在 MySQL Fuzzing 中，我们假设列和字面量都是 "T_CMP"
            # 这样允许最大限度的自由组合
            node.return_type = "T_CMP"
            return "T_CMP"
        """

        if node.node_type == 'column':
            # 【修复】：提取列的“固有类型”，拒绝被父节点上下文覆盖！
            if node.token.startswith("COL_"):
                # 例如从 "COL_INT" 中提取出 "INT"
                parts = node.token.split('_', 1)
                intrinsic_type = parts[1].upper() if len(parts) > 1 else "T_CMP"
            else:
                intrinsic_type = expected_type if expected_type != "T_ANY" else "T_CMP"
                node.token = f"COLUMN_{intrinsic_type}"

            # 强制将返回值设为自身固有类型，而不是 expected_type
            node.return_type = intrinsic_type
            return intrinsic_type


        elif node.node_type == 'literal':
            # 【修复】：提取字面量的“固有类型”，拒绝被父节点上下文覆盖！
            token_upper = str(node.token).upper()
            # Callable terminals keep their configured intrinsic type in every parent context.
            if token_upper in self.callable_terminal_tokens:
                node.token = token_upper
                node.return_type = self.get_terminal_return_type(token_upper)
                return node.return_type
            if (
                token_upper in self.configured_terminal_tokens
                and not token_upper.startswith(("COL_", "VALUE_"))
            ):
                node.token = token_upper
                node.return_type = self.get_terminal_return_type(token_upper)
                return node.return_type
            if token_upper == "NULL":
                node.token = "NULL"
                node.return_type = "T_CMP"
                return "T_CMP"
            if token_upper in {"TRUE", "FALSE"}:
                node.token = token_upper
                node.return_type = "T_BOOL"
                return "T_BOOL"
            if node.token.startswith("VALUE_"):
                if "." in node.token:
                    node.token = node.token.split(".", 1)[0]
                # 例如从 "VALUE_TYPICAL_INT" 中提取出 "INT"
                parts = node.token.split('_')
                intrinsic_type = parts[-1].upper() if len(parts) > 1 else "T_CMP"
            else:
                intrinsic_type = expected_type if expected_type != "T_ANY" else "T_CMP"

            # 强制将返回值设为自身固有类型，而不是 expected_type
            node.return_type = intrinsic_type
            return intrinsic_type

        # --- 3. 处理 Operator ---
        op_token = node.token.upper()
        rule = self.definitions.get(op_token)

        if not rule:
            # 未知算子，默认为 Bool 并接受 Any
            # print(f"Warning: Unknown operator {op_token}")
            defined_ret = "T_BOOL"
            arg_reqs = []
        else:
            defined_ret = rule[0]
            arg_reqs = rule[1]

        if op_token in {"IN", "NOT IN"}:
            defined_ret = "T_BOOL"
            arg_reqs = ["T_CMP"] * len(node.children)
        if op_token == "IS":
            defined_ret = "T_BOOL"
            arg_reqs = ["T_ANY", "T_ANY"]

        # --- 【方案B新增：兄弟节点类型启发式推导】 ---
        # 在向下递归推导之前，先“偷看”一下有没有带明确类型的兄弟节点
        specific_hint = None
        for child in node.children:
            # 1. 如果兄弟节点是 parser 包装好的类型算子 (例如 int(), float())
            if child.node_type == 'operator' and child.token.endswith('()'):
                func_name = child.token[:-2].upper()

                # 从配置中动态获取有效的数据类型
                # 如果配置中没写 valid_types，则使用通用默认类型集进行防御兜底
                valid_types = self.config.get(
                    "valid_types",
                    {"INT", "FLOAT", "DOUBLE", "DECIMAL", "STRING", "CHAR", "VARCHAR", "DATE", "TIME", "TIMESTAMP",
                     "YEAR"}
                )
                # 只有真正的数据类型强转（如 int(5)），才能作为提示。绝不采纳 FLOOR() 等计算函数！
                if func_name in valid_types:
                    specific_hint = func_name
                break
            # 2. 兼容性保护：如果兄弟节点直接是字面量 (例如 VALUE_TYPICAL_INT)
            elif child.node_type == 'literal' and child.token.startswith('VALUE_'):
                parts = child.token.split('_')
                if len(parts) >= 3:
                    specific_hint = parts[-1].upper()  # 提取后缀 INT, FLOAT 等
                break
            # --- 【新增适配 3】：如果兄弟节点是具体的抽象列 (例如 COL_INT)，提取具体类型！ ---
            elif child.node_type == 'column' and child.token.startswith('COL_'):
                parts = child.token.split('_', 1)
                if len(parts) == 2:
                    specific_hint = parts[1].upper()  # 提取后缀 INT, FLOAT 等
                break
        # ---------------------------------------------

        # 递归处理子节点
        for i, child in enumerate(node.children):
            # 获取第 i 个参数的期望类型
            if i < len(arg_reqs):
                req_child_type = arg_reqs[i]
            else:
                req_child_type = "T_ANY"  # 变长参数或越界

            # --- 【方案B新增：将偷看来的具体类型线索传给子节点】 ---
            # 如果规则原本只要求泛泛的 T_CMP，但我们从兄弟节点处找到了具体类型提示，就升级期望类型！
            if req_child_type == "T_CMP" and specific_hint:
                req_child_type = specific_hint
            # --------------------------------------------------------

            # 【递归】下发期望
            self.infer(child, req_child_type)

            # 这里可以加兼容性检查日志，但为了 Fuzzing 不报错，只做记录
            # is_compatible(actual_child_type, req_child_type) ...

        node.return_type = defined_ret
        return defined_ret
