import re  # 导入正则表达式模块，用于词法分析


# 仅用于 SQL Server 词法识别；不增加规则中的算子或终端。
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
    """保留字符串/限定名称的完整拼写，跳过注释；不执行 SQL 或加载预处理流程。"""
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
    """
    抽象语法树节点类。
    用于存储解析后的 SQL 结构片段。
    """

    def __init__(self, token, node_type, children=None):
        # 节点的显示文本，例如 "AND", ">", "col(age)"
        self.token = token
        # 节点的类型：'operator'(算子), 'column'(列), 'hole'(坑位), 'literal'(字面量), 'function'(函数)
        self.node_type = node_type
        # 子节点列表，默认为空列表
        self.children = children if children else []
        # 返回值类型 (在 Type Inference 阶段填充)，例如 "T_BOOL"
        self.return_type = None
        # 插槽类型要求 (在 Type Inference 阶段填充)
        self.slot_types = []

    def to_dict(self):
        """
        将节点序列化为字典格式，方便调试和输出 JSON。
        """
        return {
            "token": self.token,  # 节点名称
            "type": self.node_type,  # 节点类型
            "ret": self.return_type,  # 返回类型
            # 递归调用子节点的 to_dict 方法
            "children": [c.to_dict() for c in self.children]
        }


class SimpleParser:
    """
    配置驱动的简单递归下降解析器。
    用于将 SQL 谓词字符串解析为 AST。
    """

    def __init__(self, config=None):
        """
        初始化解析器。
        :param config: 包含 'precedence' (优先级) 和 'unary_operators' (一元算子) 的配置字典。
                       如果为 None，则使用内置的默认值 (MySQL风格)。
        """
        self.config = config or {}
        # --- 定义默认优先级 (MySQL 风格作为兜底方案) ---
        # 字典结构：Key 是运算符大写，Value 是优先级数值 (数值越大，结合越紧密)
        default_precedence = {
            'OR': 10, '||': 10, 'XOR': 15,  # 逻辑或
            'AND': 20, '&&': 20,  # 逻辑与
            'NOT': 25,  # 逻辑非 (低优先级)
            'BETWEEN': 30,  'NOT BETWEEN': 30,# 区间判断
            # 比较运算符，优先级相同
            '=': 40, '<=>': 40, '>=': 40, '>': 40, '<=': 40, '<': 40, '<>': 40, '!=': 40,
            'IS': 40, 'LIKE': 40, 'NOT LIKE': 40,'REGEXP': 40, 'NOT REGEXP': 40, 'IN': 40, 'NOT IN': 40, # 模式匹配和状态测试
            '|': 50, '&': 55, '<<': 60, '>>': 60,  # 位运算符
            '+': 70, '-': 70,  # 加减
            '*': 80, '/': 80, '%': 80, 'DIV': 80, 'MOD': 80,  # 乘除模
            '^': 85,  # 位异或
            'UNARY_HIGH': 100  # 高优先级一元算子 (如 !, -)
        }
        # 默认的一元运算符集合
        default_unary = {'NOT', '!', '-', '+', '~'}

        # --- 加载配置 ---
        if config:
            # 如果传入了 config，尝试从中读取 'precedence'
            # 如果配置里没有 precedence 字段，则回退到 default_precedence
            self.precedence = config.get('precedence', default_precedence)
            # 从配置读取 'unary_operators'，并转换为 set 集合以提高查找速度
            self.unary_operators = set(config.get('unary_operators', default_unary))
        else:
            # 如果没有 config，直接使用默认值
            self.precedence = default_precedence
            self.unary_operators = default_unary

        render_config = self.config.get('sql_render', {}) if isinstance(self.config, dict) else {}
        # 由传入规则识别方言，独立 parser 不依赖全局 DB_CONFIG。
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

    def tokenize(self, text): # 词法分析器 (Tokenizer)，将输入的 SQL 字符串切割成 Token 列表。
        if self.is_sqlserver:
            raw = tokenize_sqlserver(text)
            # 保留原 parser 的函数 token 格式；复合运算符来自当前规则。
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

        # --- 【新增：动态提取算子关键字，彻底消灭硬编码】 ---
        # 提取所有包含字母的算子 (如 AND, NOT LIKE, REGEXP)
        operator_tokens = set(self.precedence.keys()) | set(self.unary_operators) | set(self.operator_templates.keys())
        operator_tokens.discard('UNARY_HIGH')
        operator_tokens = [str(op) for op in operator_tokens if str(op).strip()]
        # 按长度降序排序（极其关键：保证优先匹配长的 "NOT LIKE"，而不是短的 "NOT"）
        operator_tokens.sort(key=len, reverse=True)
        # 动态构建正则片段：将空格替换为 \s+，并加上单词边界 \b
        op_patterns = []
        for op in operator_tokens:
            if re.fullmatch(r'[A-Za-z_]+(\s+[A-Za-z_]+)*', op):
                op_patterns.append(r'\b' + r'\s+'.join(re.escape(part) for part in re.split(r'\s+', op.strip())) + r'\b')
            else:
                op_patterns.append(re.escape(op))
        ops_pattern = '|'.join(op_patterns)
        # 将 SQL 特有控制关键字剥离出独立的轻量级配置（也可放入 config）
        sql_clauses = self.config.get('sql_clauses', r'\bSELECT\b|\bFROM\b|\bWHERE\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b')
        # -----------------------------------------------------

        # 定义正则表达式模式，用于匹配不同类型的 Token
        # 使用 VERBOSE 模式，允许在正则中换行和写注释
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

        tokens = []  # 初始化 Token 列表
        # 使用 re.finditer 查找所有匹配项
        # re.IGNORECASE: 忽略大小写
        # re.VERBOSE: 允许正则中包含空白和注释
        for match in re.finditer(token_pattern, text, re.IGNORECASE | re.VERBOSE):
            # 获取匹配到的字符串
            token = match.group(0).strip()
            # 如果 Token 非空，加入列表
            if token:
                tokens.append(token)
        return tokens  # 返回分词结果

    def parse(self, text):
        """
        解析主入口。
        :param text: 待解析的 SQL 谓词字符串
        :return: ASTNode 根节点，如果解析失败返回 None
        """
        # 第一步：词法分析
        self.tokens = []
        # 初始化当前 Token 指针位置
        self.pos = 0
        # 如果没有 Token，直接返回 None
        try:
            self.tokens = self.tokenize(text)
            if not self.tokens:
                return None
            # 调用递归解析逻辑，初始最低优先级为 0
            result = self.parse_expression(min_prec=0)
            if self.is_sqlserver and self.current() is not None:
                raise ValueError(f'Unconsumed SQL Server token: {self.current()}')
            return result
        except Exception as e:
            # 捕获解析过程中的异常并打印，防止程序崩溃
            print(f"Parser Error on '{text}': {e}")
            return None

    def current(self):
        """获取当前指针指向的 Token"""
        # 如果指针没有越界，返回当前 Token
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        # 否则返回 None
        return None

    def consume(self):
        """获取当前 Token 并将指针向后移动一位"""
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
        """
        递归解析表达式 (Pratt Parsing 算法核心)。
        :param min_prec: 当前上下文允许的最小优先级
        """
        # 1. 首先解析左侧的操作数 (LHS)
        # 这可能是一个原子(Atom) 或者是一个前缀一元表达式 (Unary)
        lhs = self.parse_unary()

        # 2. 循环处理紧随其后的二元运算符
        while True:
            curr = self.current()  # 查看当前 Token
            if not curr:  # 如果没有 Token 了，结束循环
                break

            op_upper = curr.upper()  # 转大写以匹配优先级字典

            # 【关键】从配置中获取当前运算符的优先级
            # 如果没找到，返回 -1 (表示它可能不是运算符，或者优先级极低)
            prec = self.precedence.get(op_upper, -1)
            op_style = self._operator_style(op_upper)
            if prec < 0 and op_style:
                prec = 40

            # 如果当前运算符优先级 低于 环境要求的最小优先级
            # 或者遇到了右括号/逗号等结束符
            # 则停止当前层级的扩展，返回 LHS
            if prec < min_prec or curr in (')', ','):
                break

            # --- 特殊处理 IN / NOT IN (变长候选列表) ---
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

            # --- 特殊处理 BETWEEN (三元运算符) ---
            if op_style == 'between':
                self.consume()  # 消费 BETWEEN
                # 解析中间部分 (AND 左边的表达式)
                middle = self.parse_expression(prec)
                # 检查是否紧跟 AND
                if self.current() and self.current().upper() == 'AND':
                    self.consume()  # 消费 AND
                    # 解析右边部分
                    rhs = self.parse_expression(prec)
                    # 构建三元节点
                    lhs = ASTNode(op_upper, 'operator', [lhs, middle, rhs])
                else:
                    if self.is_sqlserver:
                        raise ValueError('BETWEEN requires AND and an upper bound')
                    # 如果没有 AND (语法错误)，暂且构建二元节点以容错
                    lhs = ASTNode(op_upper, 'operator', [lhs, middle])
                continue  # 继续下一轮循环

            # --- 处理普通二元运算符 ---
            self.consume()  # 消费运算符

            # 递归解析运算符右侧的表达式 (RHS)
            # 这里的 min_prec 设为 prec + 1，实现了左结合律 (Left-Associative)
            rhs = self.parse_expression(prec + 1)

            # 将 LHS 和 RHS 组合成新的父节点，并更新 LHS
            lhs = ASTNode(op_upper, 'operator', [lhs, rhs])

        return lhs  # 返回最终构建的子树

    def parse_unary(self):
        """
        解析一元前缀表达式。
        例如: NOT x, -1, !flag
        """
        curr = self.current()
        if not curr:
            raise ValueError("Unexpected end")  # 异常：预期有 Token 但没了

        curr_upper = curr.upper()

        if self.is_sqlserver and curr in {'+', '-'}:
            # 有符号数值是终端，不引入规则中未定义的一元 +/- 节点。
            if self.pos + 1 < len(self.tokens) and _TSQL_NUMBER.fullmatch(self.tokens[self.pos + 1]):
                self.consume()
                return self._infer_literal_node(curr + self.consume())

        # 【关键】检查当前 Token 是否在配置的一元运算符列表中
        if curr_upper in self.unary_operators or self._operator_style(curr_upper) == 'prefix':
            self.consume()  # 消费一元运算符

            # 获取该一元运算符的优先级
            # 默认使用配置中的 UNARY_HIGH (通常很高)
            prec = self.precedence.get('UNARY_HIGH', 100)

            # 特殊处理 NOT，它的优先级通常比 ! 低
            if curr_upper == 'NOT':
                prec = self.precedence.get('NOT', 25)
            if self.is_sqlserver:
                prec = self.precedence.get(curr_upper, prec)

            # 递归解析运算符后面的操作数
            # 注意：这里递归调用 parse_expression，支持类似 !!true 的叠加
            operand = self.parse_expression(prec)

            # 构建一元算子节点
            return ASTNode(curr_upper, 'operator', [operand])

        # 如果不是一元运算符，说明是基础原子 (Atom)
        return self.parse_atom()

    def _infer_literal_node(self, token: str) -> ASTNode:
        """
        [独立解耦模块]：字面量类型推断器。
        专门负责将词法分析得到的裸 token，安全地映射为图谱所需的抽象字面量节点。
        """
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

        # 1. 识别已有的抽象常量 (例如 TYPICAL_INT)
        if token_upper.startswith("TYPICAL_") or token_upper.startswith("SINGULAR_"):
            return ASTNode(f"VALUE_{token_upper}", 'literal')

        # 2. 匹配统一的布尔和空值
        if token.lower() in ['true', 'false']:
            return ASTNode(token.upper(), 'literal')
        if token.lower() == 'null':
            return ASTNode("NULL", 'literal')

        # 3. 匹配占位符 ??
        if token == '??':
            return ASTNode("HOLE_T_CMP", 'literal')

        # 4. 匹配裸露的、变态的浮点数 (例如 .555, 00., 3., 0.0, -1.23)
        # 注意：文件开头已经 import re 了，这里绝对不要再 import re
        if re.match(r'^-?\d*\.\d*$', token) and token != '.':
            return ASTNode("VALUE_TYPICAL_FLOAT", 'literal')

        # 5. 匹配裸露的整数 (例如 123, -5, 0)
        if re.match(r'^-?\d+$', token):
            return ASTNode("VALUE_TYPICAL_INT", 'literal')

        # 6. 匹配裸露的字符串字面量 (以单双引号包裹)
        if token.startswith("'") or token.startswith('"'):
            return ASTNode("VALUE_TYPICAL_STRING", 'literal')

        # 7. 匹配乱七八糟的内部变量或遗留标识符 (如 @width..., @g333, id, ID)
        # 只要它不是我们已知的保留关键字，统统拦截，映射为通用坑位！
        if token.startswith('@') or re.match(r'^[a-zA-Z_]\w*$', token):
            if token.upper() not in ['IN', 'LIKE', 'BETWEEN', 'AND', 'OR', 'NOT', 'XOR']:
                # 凡是裸奔的字母变量，全部当成占位符
                return ASTNode("HOLE_T_CMP", 'literal')

        # 8. 终极兜底，原样返回（通常只有合法的特殊符号会走到这里）
        return ASTNode(token, 'literal')

    def parse_atom(self):
        """
        解析原子元素。
        包括: 括号表达式, 占位符, 函数调用, 列名, 字面量
        """
        token = self.consume()  # 消费当前 Token
        if token is None:
            raise ValueError('Unexpected end of expression')
        token_upper = token.upper()
        # --- 【修复：动态拦截算子裸奔，彻底消灭硬编码】 ---
        # 直接查字典：如果你是一个算子，你就绝对不能出现在变量/原子的位置！
        if token_upper in self.precedence or token_upper in self.unary_operators:
            raise ValueError(f"Operator '{token}' cannot be used as a literal/atom.")

        # 对于非算子的 SQL 控制流关键字，从动态配置中读取拦截
        forbidden_clauses = self.config.get('forbidden_clauses',
                                            {'SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE'})
        if token_upper in forbidden_clauses:
            raise ValueError(f"Illegal SQL clause keyword found: {token}")
        # ------------------------------------------------

        # PostgreSQL special expressions such as CURRENT_USER have no parentheses.
        if token_upper in self.bare_terminal_tokens:
            return ASTNode(token_upper, 'literal', [])

        # 1. 处理括号 ( )
        if token == '(':
            # 递归解析括号内的表达式，重置最小优先级为 0
            expr = self.parse_expression(0)
            # 消费闭括号 )
            if self.current() == ')':
                self.consume()
            elif self.is_sqlserver:
                raise ValueError("Expected ')' to close expression")
            return expr  # 返回括号内的表达式树

        # 2. 处理占位符 ??
        if token == '??':
            return ASTNode('??', 'hole')

        # 3. 处理函数调用或列引用 (以 '(' 结尾的 Token)
        # Tokenizer 之前已经把 "col(" 识别为一个整体 Token
        if token.endswith('(') and (not self.is_sqlserver or re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*\(', token)):
            func_name = token[:-1]  # 去掉末尾的 ( 得到函数名
            if self.is_sqlserver and func_name.lower() == 'col':
                hint = self.consume()
                if not hint or not _TSQL_IDENTIFIER.fullmatch(hint) or self.consume() != ')':
                    raise ValueError('col() requires one column identifier')
                return ASTNode('COL_STRING', 'column')
            args = []  # 初始化参数列表

            # 如果后面不是直接跟着 )，说明有参数
            if self.current() != ')':
                while True:
                    # 解析一个参数表达式
                    args.append(self.parse_expression(0))
                    # 如果遇到逗号，消费逗号并继续解析下一个参数
                    if self.current() == ',':
                        self.consume()
                    else:
                        break  # 没有逗号，参数列表结束

            # 消费闭括号 )
            if self.current() == ')':
                self.consume()
            elif self.is_sqlserver:
                raise ValueError("Expected ')' to close function")

            # ==================== 【列名与常量的抽象化逻辑】 ====================
            func_lower = func_name.lower()
            function_token = f"{func_name.upper()}()"

            if self.is_sqlserver:
                wrapper_type = _TSQL_DSL_TYPES.get(func_lower)
                if wrapper_type:
                    if len(args) != 1:
                        raise ValueError(f'{func_name} DSL wrapper requires one expression')
                    # 字面量包装只改终端类别；不能抹去内部真实函数/算子及其边。
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

            # Configured zero-argument SQL functions are callable terminals.
            if not args and function_token in self.callable_terminal_tokens:
                return ASTNode(function_token, 'literal', [])

            # 1. 抽象化 Column (列)
            if func_lower == 'col':
                # 丢弃具体的列名 (如 age, id)，将其规约为纯抽象的无子节点终端
                return ASTNode("col()", 'column', [])

            # 2. 抽象化 Value (常量)
            type_wrappers = ['int', 'float', 'decimal', 'double', 'string', 'char', 'varchar', 'date', 'timestamp',
                             'time', 'hex', 'bit']
            if func_lower in type_wrappers and len(args) == 1 and args[0].node_type == 'literal':
                val_str = args[0].token.strip("'\"").lower()
                is_singular = False

                # 判断是否为 Singular Value (奇异/边界值)
                if func_lower in ['int', 'float', 'decimal', 'double']:
                    if val_str in ['0', '0.0', '-1', '1', '2147483647', '-2147483648', '999999.99', '-0.0']:
                        is_singular = True
                elif func_lower in ['string', 'char', 'varchar']:
                    if val_str in ['', ' ', '\\0', 'null']:
                        is_singular = True

                val_type = "SINGULAR" if is_singular else "TYPICAL"
                abstract_token = f"VALUE_{val_type}_{func_name.upper()}"

                # 构造抽象的 VALUE 节点，替代原有的具体数字
                abstract_node = ASTNode(abstract_token, 'literal', [])
                return ASTNode(function_token, 'operator', [abstract_node])

            # 3. 默认情况：保留普通函数及其参数
            node_type = 'operator'
            return ASTNode(function_token, node_type, args)
            # =================================================================

            """
            # 根据函数名判断节点类型
            # 如果是 'col'，标记为 'column'，否则标记为 'operator'
            node_type = 'column' if func_name.lower() == 'col' else 'operator'

            # 构建节点，Token 还原为带括号的形式以便显示
            return ASTNode(token + ")", node_type, args)
            """
        # 4. 默认情况：视为字面量 (Literal)
        token_upper = token.upper()

        # 识别 pkg_manager 传来的抽象列名 (例如 col_int)
        if token_upper.startswith("COL_") and (not self.is_sqlserver or re.fullmatch(r'COL_[A-Z0-9_]+', token_upper)):
            return ASTNode(token_upper, 'column')

        if self.is_sqlserver and token in {')', ','}:
            raise ValueError(f'Unexpected SQL Server atom: {token}')

        # 将裸奔的 token 交给专属推断器处理
        return self._infer_literal_node(token)
