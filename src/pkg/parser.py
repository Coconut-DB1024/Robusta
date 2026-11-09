import re  


_TSQL_NAME_PART = r'(?:\[(?:[^\]]|\]\])+\]|"(?:[^"]|"")+"|(?:[^\W\d]|[_#@])[\w$#@]*)'
_TSQL_IDENTIFIER = re.compile(_TSQL_NAME_PART + r'(?:\s*\.\s*' + _TSQL_NAME_PART + r')*')
_TSQL_STRING = re.compile(r"[Nn]?'(?:[^']|'')*'")
_TSQL_NUMBER = re.compile(r'(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?')
_TSQL_DSL_TYPES = {
    **dict.fromkeys(['int', 'integer', 'tinyint', 'smallint', 'bigint', 'bit'], 'INT'),
    **dict.fromkeys(['float', 'double', 'decimal', 'numeric', 'real', 'money', 'smallmoney'], 'FLOAT'),
    **dict.fromkeys(['string', 'varchar', 'nvarchar', 'text', 'ntext'], 'STRING'),
    **dict.fromkeys(['binary', 'varbinary', 'hex', 'bin', 'timestamp', 'rowversion', 'image'], 'BINARY'),
    **dict.fromkeys(['date', 'time', 'datetime', 'datetime2', 'datetimeoffset', 'smalldatetime'], 'DATE'),
    'uniqueidentifier': 'UNIQUEIDENTIFIER',
}


def tokenize_sqlserver(text):
    tokens, i = [], 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        if text.startswith('--', i):
            end = text.find('\n', i + 2)
            i = len(text) if end < 0 else end + 1
            continue
        if text.startswith('/*', i):
            depth, i = 1, i + 2
            while i < len(text) and depth:
                if text.startswith('/*', i):
                    depth, i = depth + 1, i + 2
                elif text.startswith('*/', i):
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
            if depth:
                raise ValueError('Unterminated SQL Server comment')
            continue
        match = _TSQL_STRING.match(text, i) or _TSQL_IDENTIFIER.match(text, i)
        if match:
            tokens.append(match.group())
            i = match.end()
            continue
        if text[i] in "'\"[":
            raise ValueError('Unterminated SQL Server string or identifier')
        if text[i:i + 2].lower() == '0x':
            match = re.match(r'0[xX][0-9A-Za-z_]*', text[i:])
            token = match.group()
            if not re.fullmatch(r'0[xX][0-9a-fA-F]*', token):
                raise ValueError('Invalid SQL Server binary literal')
            tokens.append(token)
            i += len(token)
            continue
        match = _TSQL_NUMBER.match(text, i)
        if match:
            tokens.append(match.group())
            i = match.end()
            continue
        token = text[i:i + 2]
        if token not in {'<=', '>=', '<>', '!=', '!<', '!>', '<<', '>>', '??'}:
            token = text[i]
        if token not in {'<=', '>=', '<>', '!=', '!<', '!>', '<<', '>>', '??'} and token not in '+-*/%=<>&|^~(),':
            raise ValueError(f'Unexpected SQL Server token: {token}')
        tokens.append(token)
        i += len(token)
    return tokens


class ASTNode:
    def __init__(self, token, node_type, children=None):
        self.token = token

        self.node_type = node_type

        self.children = children if children else []

        self.return_type = None

        self.slot_types = []

    def to_dict(self):
        return {
            "token": self.token,  
            "type": self.node_type,  
            "ret": self.return_type,  

            "children": [c.to_dict() for c in self.children]
        }


class SimpleParser:
    def __init__(self, config=None):
        self.config = config or {}

        default_precedence = {
            'OR': 10, '||': 10, 'XOR': 15,  
            'AND': 20, '&&': 20,  
            'NOT': 25,  
            'BETWEEN': 30,  'NOT BETWEEN': 30,

            '=': 40, '<=>': 40, '>=': 40, '>': 40, '<=': 40, '<': 40, '<>': 40, '!=': 40,
            'IS': 40, 'LIKE': 40, 'NOT LIKE': 40,'REGEXP': 40, 'NOT REGEXP': 40, 'IN': 40, 'NOT IN': 40, 
            '|': 50, '&': 55, '<<': 60, '>>': 60,  
            '+': 70, '-': 70,  
            '*': 80, '/': 80, '%': 80, 'DIV': 80, 'MOD': 80,  
            '^': 85,  
            'UNARY_HIGH': 100  
        }

        default_unary = {'NOT', '!', '-', '+', '~'}

        if config:
            self.precedence = config.get('precedence', default_precedence)

            self.unary_operators = set(config.get('unary_operators', default_unary))
        else:
            self.precedence = default_precedence
            self.unary_operators = default_unary

        render_config = self.config.get('sql_render', {}) if isinstance(self.config, dict) else {}

        self.is_sqlserver = str(render_config.get('version', '')).startswith('sqlserver_')
        self.operator_templates = render_config.get('operator_templates', {}) if isinstance(render_config, dict) else {}
        self.operator_styles = {
            str(token).upper(): str(template.get('style', '')).strip().lower()
            for token, template in self.operator_templates.items()
            if isinstance(template, dict)
        }
        bitmap_config = self.config.get('candidate_bitmap', {}) if isinstance(self.config, dict) else {}
        callable_tokens = bitmap_config.get('callable_terminal_tokens', []) if isinstance(bitmap_config, dict) else []
        self.callable_terminal_tokens = {
            str(item).strip().upper()
            for item in callable_tokens
            if str(item).strip()
        }
        terminal_tokens = bitmap_config.get('terminal_tokens', {}) if isinstance(bitmap_config, dict) else {}
        self.bare_terminal_tokens = {
            str(item).strip().upper()
            for item in terminal_tokens.keys()
            if str(item).strip()
            and not str(item).strip().endswith('()')
            and not str(item).strip().upper().startswith(('COL_', 'VALUE_'))
        }

    def _operator_style(self, token):
        return self.operator_styles.get(str(token).upper(), "")

    def tokenize(self, text): 
        if self.is_sqlserver:
            raw = tokenize_sqlserver(text)

            compounds = sorted((op.split() for op in self.precedence if ' ' in op), key=len, reverse=True)
            keywords = set(self.config.get('non_function_keywords', []))
            tokens, i = [], 0
            while i < len(raw):
                compound = next((parts for parts in compounds
                                 if [t.upper() for t in raw[i:i + len(parts)]] == parts), None)
                if compound:
                    tokens.append(' '.join(compound))
                    i += len(compound)
                elif (i + 1 < len(raw) and raw[i + 1] == '('
                      and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', raw[i])
                      and raw[i].upper() not in keywords):
                    tokens.append(raw[i] + '(')
                    i += 2
                else:
                    tokens.append(raw[i])
                    i += 1
            return tokens

        operator_tokens = set(self.precedence.keys()) | set(self.unary_operators) | set(self.operator_templates.keys())
        operator_tokens.discard('UNARY_HIGH')
        operator_tokens = [str(op) for op in operator_tokens if str(op).strip()]

        operator_tokens.sort(key=len, reverse=True)

        op_patterns = []
        for op in operator_tokens:
            if re.fullmatch(r'[A-Za-z_]+(\s+[A-Za-z_]+)*', op):
                op_patterns.append(r'\b' + r'\s+'.join(re.escape(part) for part in re.split(r'\s+', op.strip())) + r'\b')
            else:
                op_patterns.append(re.escape(op))
        ops_pattern = '|'.join(op_patterns)

        sql_clauses = self.config.get('sql_clauses', r'\bSELECT\b|\bFROM\b|\bWHERE\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b')

        token_pattern = rf"""
            ({ops_pattern}|{sql_clauses})|     # 1. 【修改这里】：使用动态插入的合法关键字组合
            ([a-zA-Z_][a-zA-Z0-9_]*\s*\()|     # 2. 匹配函数调用或列名，例如 "col(" 或 "abs("
            (\?\?)|                            # 2. 匹配占位符 "??"
            (<=>|<<|>>|>=|<=|!=|<>|&&|\|\|)|   # 3. 匹配多字符运算符 (注意顺序，长的在前)
            (['"].*?['"])|                     # 5. 匹配字符串字面量 (单引号或双引号)
            ([0-9]+\.?[0-9]*)|                 # 6. 匹配数字 (整数或浮点数)
            ([+\-*/%=!<>&|^~(),])|             # 7. 匹配单字符运算符
            ([^+\-*/%=!<>&|^~(),\s]+)          # 8. 兜底匹配：匹配除了特殊符号和空白之外的任何内容
        """

        tokens = []  

        for match in re.finditer(token_pattern, text, re.IGNORECASE | re.VERBOSE):
            token = match.group(0).strip()

            if token:
                tokens.append(token)
        return tokens  

    def parse(self, text):
        self.tokens = []

        self.pos = 0

        try:
            self.tokens = self.tokenize(text)
            if not self.tokens:
                return None

            result = self.parse_expression(min_prec=0)
            if self.is_sqlserver and self.current() is not None:
                raise ValueError(f'Unconsumed SQL Server token: {self.current()}')
            return result
        except Exception as e:
            print(f"Parser Error on '{text}': {e}")
            return None

    def current(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]

        return None

    def consume(self):
        token = self.current()
        self.pos += 1
        return token

    def parse_expression_list_until_rparen(self):
        if self.current() != '(':
            return [self.parse_expression(0)]

        self.consume()
        items = []

        while True:
            if self.current() is None:
                raise ValueError("Expected ')' to close expression list")
            if self.current() == ')':
                break

            items.append(self.parse_expression(0))

            if self.current() == ',':
                self.consume()
                if self.is_sqlserver and self.current() == ')':
                    raise ValueError('Trailing comma in SQL Server expression list')
                continue
            if self.current() == ')':
                break
            raise ValueError(f"Expected ',' or ')' in expression list, got {self.current()}")

        if self.current() != ')':
            raise ValueError("Expected ')' to close expression list")
        self.consume()
        return items

    def parse_expression(self, min_prec):
        lhs = self.parse_unary()

        while True:
            curr = self.current()  
            if not curr:  
                break

            op_upper = curr.upper()  

            prec = self.precedence.get(op_upper, -1)
            op_style = self._operator_style(op_upper)
            if prec < 0 and op_style:
                prec = 40

            if prec < min_prec or curr in (')', ','):
                break

            op_style = self._operator_style(op_upper)

            if op_style == 'postfix':
                self.consume()
                lhs = ASTNode(op_upper, 'operator', [lhs])
                continue

            if op_style == 'in_list':
                self.consume()
                if self.is_sqlserver and self.current() != '(':
                    raise ValueError(f'{op_upper} requires a parenthesized list')
                list_items = self.parse_expression_list_until_rparen()
                if not list_items:
                    raise ValueError(f"{op_upper} requires at least one list item")
                lhs = ASTNode(op_upper, 'operator', [lhs] + list_items)
                continue

            if op_style == 'between':
                self.consume()  

                middle = self.parse_expression(prec)

                if self.current() and self.current().upper() == 'AND':
                    self.consume()  

                    rhs = self.parse_expression(prec)

                    lhs = ASTNode(op_upper, 'operator', [lhs, middle, rhs])
                else:
                    if self.is_sqlserver:
                        raise ValueError('BETWEEN requires AND and an upper bound')

                    lhs = ASTNode(op_upper, 'operator', [lhs, middle])
                continue  

            self.consume()  

            rhs = self.parse_expression(prec + 1)

            lhs = ASTNode(op_upper, 'operator', [lhs, rhs])

        return lhs  

    def parse_unary(self):
        curr = self.current()
        if not curr:
            raise ValueError("Unexpected end")  

        curr_upper = curr.upper()

        if self.is_sqlserver and curr in {'+', '-'}:
            if self.pos + 1 < len(self.tokens) and _TSQL_NUMBER.fullmatch(self.tokens[self.pos + 1]):
                self.consume()
                return self._infer_literal_node(curr + self.consume())

        if curr_upper in self.unary_operators or self._operator_style(curr_upper) == 'prefix':
            self.consume()  

            prec = self.precedence.get('UNARY_HIGH', 100)

            if curr_upper == 'NOT':
                prec = self.precedence.get('NOT', 25)
            if self.is_sqlserver:
                prec = self.precedence.get(curr_upper, prec)

            operand = self.parse_expression(prec)

            return ASTNode(curr_upper, 'operator', [operand])

        return self.parse_atom()

    def _infer_literal_node(self, token: str) -> ASTNode:
        token_upper = token.upper()

        if self.is_sqlserver:
            if _TSQL_STRING.fullmatch(token):
                return ASTNode('VALUE_TYPICAL_STRING', 'literal')
            if re.fullmatch(r'0[xX][0-9a-fA-F]*', token):
                return ASTNode('VALUE_TYPICAL_BINARY', 'literal')
            if _TSQL_NUMBER.fullmatch(token.lstrip('+-')):
                kind = 'FLOAT' if any(c in token.lower() for c in '.e') else 'INT'
                return ASTNode(f'VALUE_TYPICAL_{kind}', 'literal')
            if re.fullmatch(r'VALUE_[A-Z0-9_]+', token_upper):
                return ASTNode(token_upper, 'literal')
            if token.startswith('@'):
                return ASTNode('HOLE_T_CMP', 'literal')
            if _TSQL_IDENTIFIER.fullmatch(token) and not token_upper.startswith(('TYPICAL_', 'SINGULAR_', 'HOLE_')):
                return ASTNode('COL_STRING', 'column')

        if token_upper.startswith("TYPICAL_") or token_upper.startswith("SINGULAR_"):
            return ASTNode(f"VALUE_{token_upper}", 'literal')

        if token.lower() in ['true', 'false']:
            return ASTNode(token.upper(), 'literal')
        if token.lower() == 'null':
            return ASTNode("NULL", 'literal')

        if token == '??':
            return ASTNode("HOLE_T_CMP", 'literal')

        if re.match(r'^-?\d*\.\d*$', token) and token != '.':
            return ASTNode("VALUE_TYPICAL_FLOAT", 'literal')

        if re.match(r'^-?\d+$', token):
            return ASTNode("VALUE_TYPICAL_INT", 'literal')

        if token.startswith("'") or token.startswith('"'):
            return ASTNode("VALUE_TYPICAL_STRING", 'literal')

        if token.startswith('@') or re.match(r'^[a-zA-Z_]\w*$', token):
            if token.upper() not in ['IN', 'LIKE', 'BETWEEN', 'AND', 'OR', 'NOT', 'XOR']:
                return ASTNode("HOLE_T_CMP", 'literal')

        return ASTNode(token, 'literal')

    def parse_atom(self):
        token = self.consume()  
        if token is None:
            raise ValueError('Unexpected end of expression')
        token_upper = token.upper()

        if token_upper in self.precedence or token_upper in self.unary_operators:
            raise ValueError(f"Operator '{token}' cannot be used as a literal/atom.")

        forbidden_clauses = self.config.get('forbidden_clauses',
                                            {'SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE'})
        if token_upper in forbidden_clauses:
            raise ValueError(f"Illegal SQL clause keyword found: {token}")

        if token_upper in self.bare_terminal_tokens:
            return ASTNode(token_upper, 'literal', [])

        if token == '(':
            expr = self.parse_expression(0)

            if self.current() == ')':
                self.consume()
            elif self.is_sqlserver:
                raise ValueError("Expected ')' to close expression")
            return expr  

        if token == '??':
            return ASTNode('??', 'hole')

        if token.endswith('(') and (not self.is_sqlserver or re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*\(', token)):
            func_name = token[:-1]  
            if self.is_sqlserver and func_name.lower() == 'col':
                hint = self.consume()
                if not hint or not _TSQL_IDENTIFIER.fullmatch(hint) or self.consume() != ')':
                    raise ValueError('col() requires one column identifier')
                return ASTNode('COL_STRING', 'column')
            args = []  

            if self.current() != ')':
                while True:
                    args.append(self.parse_expression(0))

                    if self.current() == ',':
                        self.consume()
                    else:
                        break  

            if self.current() == ')':
                self.consume()
            elif self.is_sqlserver:
                raise ValueError("Expected ')' to close function")

            func_lower = func_name.lower()
            function_token = f"{func_name.upper()}()"

            if self.is_sqlserver:
                wrapper_type = _TSQL_DSL_TYPES.get(func_lower)
                if wrapper_type:
                    if len(args) != 1:
                        raise ValueError(f'{func_name} DSL wrapper requires one expression')

                    child = args[0]
                    if child.node_type == 'literal' and child.token.startswith('VALUE_'):
                        child.token = f'VALUE_TYPICAL_{wrapper_type}'
                    return child
                if not args and function_token in self.callable_terminal_tokens:
                    return ASTNode(function_token, 'literal', [])
                definition = self.config.get('definitions', {}).get(function_token)
                if not definition or len(args) != len(definition[1]):
                    raise ValueError(f'Function outside SQL Server rule scope or wrong arity: {func_name}')
                return ASTNode(function_token, 'operator', args)

            if not args and function_token in self.callable_terminal_tokens:
                return ASTNode(function_token, 'literal', [])

            if func_lower == 'col':
                return ASTNode("col()", 'column', [])

            type_wrappers = ['int', 'float', 'decimal', 'double', 'string', 'char', 'varchar', 'date', 'timestamp',
                             'time', 'hex', 'bit']
            if func_lower in type_wrappers and len(args) == 1 and args[0].node_type == 'literal':
                val_str = args[0].token.strip("'\"").lower()
                is_singular = False

                if func_lower in ['int', 'float', 'decimal', 'double']:
                    if val_str in ['0', '0.0', '-1', '1', '2147483647', '-2147483648', '999999.99', '-0.0']:
                        is_singular = True
                elif func_lower in ['string', 'char', 'varchar']:
                    if val_str in ['', ' ', '\\0', 'null']:
                        is_singular = True

                val_type = "SINGULAR" if is_singular else "TYPICAL"
                abstract_token = f"VALUE_{val_type}_{func_name.upper()}"

                abstract_node = ASTNode(abstract_token, 'literal', [])
                return ASTNode(function_token, 'operator', [abstract_node])

            node_type = 'operator'
            return ASTNode(function_token, node_type, args)

        token_upper = token.upper()

        if token_upper.startswith("COL_") and (not self.is_sqlserver or re.fullmatch(r'COL_[A-Z0-9_]+', token_upper)):
            return ASTNode(token_upper, 'column')

        if self.is_sqlserver and token in {')', ','}:
            raise ValueError(f'Unexpected SQL Server atom: {token}')

        return self._infer_literal_node(token)
