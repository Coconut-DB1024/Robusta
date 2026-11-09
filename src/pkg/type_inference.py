class TypeInferencer:
    def __init__(self, config):
        self.config = config or {}  
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
        if raw_type is None:
            return "T_ANY"

        text = str(raw_type).strip().upper()
        if not text:
            return "T_ANY"

        return str(self.coarse_type_aliases.get(text, text)).upper()

    def get_operator_signature(self, token):
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
        if not node:
            return None

        if node.node_type == 'hole':
            target_type = expected_type if expected_type != "T_ANY" else "T_CMP"

            node.token = f"HOLE_{target_type}"
            node.return_type = target_type
            return target_type

        if node.node_type == 'column':
            if node.token.startswith("COL_"):
                parts = node.token.split('_', 1)
                intrinsic_type = parts[1].upper() if len(parts) > 1 else "T_CMP"
            else:
                intrinsic_type = expected_type if expected_type != "T_ANY" else "T_CMP"
                node.token = f"COLUMN_{intrinsic_type}"

            node.return_type = intrinsic_type
            return intrinsic_type

        elif node.node_type == 'literal':
            token_upper = str(node.token).upper()

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

                parts = node.token.split('_')
                intrinsic_type = parts[-1].upper() if len(parts) > 1 else "T_CMP"
            else:
                intrinsic_type = expected_type if expected_type != "T_ANY" else "T_CMP"

            node.return_type = intrinsic_type
            return intrinsic_type

        op_token = node.token.upper()
        rule = self.definitions.get(op_token)

        if not rule:
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

        specific_hint = None
        for child in node.children:
            if child.node_type == 'operator' and child.token.endswith('()'):
                func_name = child.token[:-2].upper()

                valid_types = self.config.get(
                    "valid_types",
                    {"INT", "FLOAT", "DOUBLE", "DECIMAL", "STRING", "CHAR", "VARCHAR", "DATE", "TIME", "TIMESTAMP",
                     "YEAR"}
                )

                if func_name in valid_types:
                    specific_hint = func_name
                break

            elif child.node_type == 'literal' and child.token.startswith('VALUE_'):
                parts = child.token.split('_')
                if len(parts) >= 3:
                    specific_hint = parts[-1].upper()  
                break

            elif child.node_type == 'column' and child.token.startswith('COL_'):
                parts = child.token.split('_', 1)
                if len(parts) == 2:
                    specific_hint = parts[1].upper()  
                break

        for i, child in enumerate(node.children):
            if i < len(arg_reqs):
                req_child_type = arg_reqs[i]
            else:
                req_child_type = "T_ANY"  

            if req_child_type == "T_CMP" and specific_hint:
                req_child_type = specific_hint

            self.infer(child, req_child_type)

        node.return_type = defined_ret
        return defined_ret
