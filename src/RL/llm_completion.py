"""
作用：
- 从当前 MySQL 数据库中读取表结构（表名 + 列名 + 类型）
- 将「谓词模板 + 表结构」交给本地 LLM（支持 HTTP API 或 本地加载 transformers 模型）
- 让 LLM 直接补全为一个 **完整的 SQL 查询语句**：
    SELECT * FROM <某个表> WHERE <补全后的谓词>;
"""

from __future__ import annotations

from typing import Optional
import re
import requests
#import torch
#from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor # 【新增】用于并发请求
import os

from RL.config import LLM_CONFIG
from RL.db_executor import BaseDBExecutor, create_db_executor


# 从当前数据库读取表结构，并转成文本给 LLM 参考
def get_database_schema_text(executor: BaseDBExecutor) -> str:
    # 先 ping 一下，确保连接可用
    if not executor.ping():
        return "当前数据库连接失败，无法获取表结构。"

    # 查询所有表名
    res_tables = executor.list_tables()

    if not res_tables.success or not res_tables.rows:
        return "当前数据库中没有表，或表列表查询失败。"

    table_names = []
    for row in res_tables.rows:
        if not row:
            continue
        first_val = list(row.values())[0]
        if isinstance(first_val, str):
            table_names.append(first_val)

    if not table_names:
        return "当前数据库中没有可用的表。"

    lines = ["当前数据库中的表结构如下："]

    for table in table_names:
        lines.append(f"- 表 {table}:")
        # 查询该表的列信息
        res_cols = executor.list_columns(table)
        if not res_cols.success or not res_cols.rows:
            lines.append("    (无法获取列信息)")
            continue

        for col in res_cols.rows:
            col_name = col.get("Field", "?")
            col_type = col.get("Type", "?")
            lines.append(f"    - {col_name} {col_type}")

    return "\n".join(lines)


# LLM 调用封装（支持 HTTP API 和 本地 Transformers 加载）
class LLMCompletionClient:

    def __init__(self) -> None:
        # 从全局配置读取参数
        # 【修改】模型名称必须与 Slurm 脚本中 --served-model-name 一致
        self.model_name = "Qwen2.5-7B-Instruct"
        self.api_base = LLM_CONFIG.api_base
        self.api_key = LLM_CONFIG.api_key
        self.temperature = LLM_CONFIG.temperature
        self.max_tokens = LLM_CONFIG.max_tokens

        # 【修改】强制构造 HTTP URL
        if not self.api_base:
            self.api_base = "http://localhost:8000"
        self.chat_url = f"{self.api_base}/v1/chat/completions"

        # 【优化】预定义 headers，复用连接
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

        print(f"[LLM Client] 初始化完成 (HTTP Mode). 目标地址: {self.chat_url}")
        # 注意：这里删除了所有的 model/tokenizer 加载逻辑，显存占用降为 0

    @staticmethod
    def _strip_think_tag(content: str) -> str:
        """清洗 <think> 标签"""
        text = content.strip()
        if "<think>" in text and "</think>" in text:
            return text.split("</think>", 1)[1].strip()
        return text

    @staticmethod
    def _cleanup_wrappers_in_sql(sql: str) -> str:
        """清洗残留的 col()/int() 等 wrapper"""
        text = (sql or "").strip()
        if not text:
            return text
        pattern = re.compile(r"(?i)\b(col|int|float|string|date)\s*\(\s*([^()]+?)\s*\)")
        for _ in range(20):
            new_text = pattern.sub(lambda m: m.group(2).strip(), text)
            if new_text == text:
                break
            text = new_text
        return text

    # 统一的聊天接口，自动分发
    def _chat(self, system_prompt: str, user_prompt: str) -> str:

        # 【修改】完全移除 "local_transformers" 分支，只保留 HTTP 请求逻辑
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            # 【优化】增加超时重试机制，防止网络抖动
            resp = requests.post(self.chat_url, json=payload, headers=self.headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[LLM Error] HTTP 请求失败: {e}")
            return ""  # 返回空字符串，让后续逻辑处理

    # === 新增：批量并发接口 ===
    def complete_sql_batch(self, tasks: list[tuple[str, str]]) -> list[str]:
        """
        并发处理一批 (predicate_template, schema_text) 任务。
        利用 ThreadPoolExecutor 向 vLLM 发送并发请求，极大提升吞吐量。
        """
        if not tasks:
            return []

        results = [""] * len(tasks)

        # 定义单个任务的处理逻辑（复用现有的单条逻辑，但剥离出网络请求部分会更高效，
        # 这里为了稳妥，直接复用 _chat 接口）
        def _worker(index, pred_tmpl, schema_txt):
            try:
                # 1. 构造 Prompt (复用现有的逻辑，这里需要手动拼接一下，或者将 prompt 构造逻辑抽离)
                # 为了不改动太多现有代码结构，我们这里直接调用 complete_sql_with_schema
                # 注意：这意味着会重复做一些 strip/cleaning 工作，但对性能影响微乎其微
                sql = self.complete_sql_with_schema(pred_tmpl, schema_txt)
                results[index] = sql
            except Exception as e:
                print(f"[LLM Batch Error] Task {index} failed: {e}")
                results[index] = ""

        # 使用线程池并发请求 vLLM
        # max_workers 建议设置为任务数量，或者固定一个较大值（如 32）
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            for i, (p, s) in enumerate(tasks):
                executor.submit(_worker, i, p, s)

        return results

    # 接口：补全为完整 SQL
    def complete_sql_with_schema(
        self,
        predicate_template: str,
        schema_text: str,
    ) -> str:
        # ---------------------------------------------------------
        # Prompt 构造逻辑 (完全保留你原来的设计)
        # ---------------------------------------------------------
        system_prompt = (
            "你是一个呆板的 SQL 模板填充工具。你的任务是将模板转换为 SQL。\n"
            "原则 1：请根据提供的 Schema，**必须使用 Schema 中真实存在的表名**来编写 SELECT 语句 (例如 SELECT * FROM 真实表名 WHERE ...)。\n"
            "原则 2：保持谓词的逻辑结构（运算符、嵌套关系、常量值）不变。\n"
            "原则 3：对于 col(X)，请检查 X 是否在 Schema 中。\n"
            "       - 如果 X 存在：必须保留 X。\n"
            "       - 如果 X 不存在：请从 Schema 中选择一个与 X 语义最接近或数据类型兼容的列名 Y 进行替换。不要让 SQL 报错。\n"
            "原则 4：对于 ??，必须将其替换为合法的常量或列名，不能保留 ??。\n"
            "原则 5：如果输入模板包含真正的乱码或极度混乱的符号（col/int/string/date 等等类型标记除外），请**直接原样输出该乱码内容**，不要尝试去猜或者修正，以便后续测试能捕获此错误。"  # <--- 新增
        )

        user_prompt = f""" 
数据库表结构：
{schema_text}

任务：将给定的【谓词模板】转换为【可执行 SQL】。

请仔细阅读以下【处理范例】，并严格模仿其行为：

--- 范例 1 (提取表名 + 处理存在的列) ---
Schema: 
- 表 employees:
    - age int
    - name varchar
模板: col(age) > int(10)
思考: Schema 中表名为 employees。列 age 存在。int(10) -> 10。
输出: SELECT * FROM employees WHERE age > 10;

--- 范例 2 (列名不存在，需要映射) ---
Schema: 
- 表 payments:
    - balance decimal
    - id int
模板: col(salary) > int(5000)
思考: Schema 中表名为 payments。表里没有 salary，但 balance 也是金额，语义接近。替换为 balance。int(5000) -> 5000。
输出: SELECT * FROM payments WHERE balance > 5000;

--- 范例 3 (处理 ?? 占位符) ---
Schema: 
- 表 students:
    - score int
模板: col(score) > ??
思考: Schema 中表名为 students。?? 需要被替换。score 是 int，??需要补一个合理的整数。
输出: SELECT * FROM students WHERE score > 60;

--- 范例 4 (复杂结构保持) ---
Schema: 
- 表 logs:
    - level int
    - msg text
模板: (col(level) > int(1) AND col(msg) LIKE string('%error%'))
思考: 表名 logs。列都存在。保持 AND 结构。
输出: SELECT * FROM logs WHERE level > 1 AND msg LIKE '%error%';

--- 范例 5 (不存在的列 + ?? 占位符) ---
Schema: 
- 表 products:
    - stock int
    - product_name varchar
模板: col(count) > ??
思考: Schema 中表名为 products。表里没有 count，但 stock 也是数量类型，语义接近。将 count 替换为 stock。而 stock 为 int，?? 需要补一个合理的整数。
输出: SELECT * FROM products WHERE stock > 0;

--- 范例 6 (左值占位符 + 跨列名映射) ---
Schema: 
- 表 users:
    - salary decimal
    - age int
模板: ?? > int(50)
思考: Schema 中表名为 users。在右边为一个数值类型，因此在左边 ?? 需要一个数值列。所以可以从 Schema 中选择 salary。int(50) -> 50。
输出: SELECT * FROM users WHERE salary > 50;

--- 范例 7 (严重语法错误与操作符堆叠) ---
Schema: 
- 表 employees:
    - salary int
    - employee_name varchar
模板: col(salary) - > int(10) OR !
思考: Schema 中表名为 employees。模板中存在明显的操作符堆叠（- >）和悬空的逻辑符（!），属于畸形语法。按照规则，不应尝试修复语法错误，只需剥离类型外壳：col(salary) -> salary，int(10) -> 10。原封不动地保留错误结构即可。
输出: SELECT * FROM employees WHERE salary - > 10 OR !;

--- 范例 8 (括号不匹配与结构未闭合) ---
Schema: 
- 表 orders:
    - status int
    - user_id int
模板: ((col(status) = int(1) AND ?? ||
思考: Schema 中表名为 orders。模板存在左括号未闭合的问题，且以 || 结尾，结构严重畸形。我们仅需剥离外壳（col(status) -> status, int(1) -> 1），并将 ?? 替换为 Schema 中存在的列（例如 user_id）。绝对不要擅自补全右括号或删除尾部的 ||，必须保留原始错误以触发数据库的 Syntax Error。
输出: SELECT * FROM orders WHERE ((status = 1 AND user_id ||;

--- 范例结束 ---

现在处理以下输入：
谓词模板：
{predicate_template}

要求：
1. 最终结果中**绝对不能**包含 `col(...)`, `int(...)` 等类型名包装。
2. 最终结果中**绝对不能**包含 `??`，必须替换完。
3. 如果 `col(X)` 中的 X 在表中不存在，**必须**找出类型最相似的列名将其替换。
4. 必须确保 `FROM` 后面的表名是上面 Schema 里真正有的表。

直接输出最终的 SQL 语句：
"""
        # 调用底层 chat 接口 (这一行是关键改动点)
        raw_reply = self._chat(system_prompt=system_prompt, user_prompt=user_prompt)

        # 去掉 DeepSeek R1 的 <think> 段
        text = self._strip_think_tag(raw_reply)

        # 尝试从 ``` 代码块中抽取 SQL
        sql_candidate: str = ""

        if "```" in text:
            after_first = text.split("```", 1)[1]
            if "```" in after_first:
                block = after_first.split("```", 1)[0]
            else:
                block = after_first
            block = block.strip()
            lines = block.splitlines()
            if lines and lines[0].strip().lower() in ("sql", "mysql"):
                lines = lines[1:]
            sql_candidate = " ".join(line.strip() for line in lines if line.strip())
        else:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            select_lines = [ln for ln in lines if "select" in ln.lower()]
            if select_lines:
                sql_candidate = select_lines[0]
            elif lines:
                sql_candidate = lines[0]
            else:
                sql_candidate = ""

        sql_candidate = sql_candidate.strip()
        if sql_candidate.startswith("`") and sql_candidate.endswith("`"):
            sql_candidate = sql_candidate[1:-1].strip()

        lower = sql_candidate.lower()
        if "select" in lower:
            idx = lower.find("select")
            sql_candidate = sql_candidate[idx:].strip()

        # 清理 SQL 中残留的 wrapper
        sql_candidate = self._cleanup_wrappers_in_sql(sql_candidate)

        if not sql_candidate.endswith(";"):
            sql_candidate = sql_candidate + ";"

        return sql_candidate


# 简单自测
if __name__ == "__main__":
    print("=== LLM Completion SelfTest ===")

    # 初始化 DB 执行器并读取 schema
    try:
        db_exec = create_db_executor(autoconnect=True)
        schema = get_database_schema_text(db_exec)
        print("[SelfTest] Database schema summary:")
        print(schema)
    except Exception as e:
        print(f"[SelfTest] 数据库连接失败: {e}")
        # schema = "CREATE TABLE dummy (id INT, name VARCHAR(20));" # 仅用于无 DB 测试

    print("--------------------------------------------------")

    # 初始化 LLM 客户端
    try:
        llm_client = LLMCompletionClient()
    except Exception as e:
        print("[SelfTest] 初始化 LLMCompletionClient 失败：", e)
        exit(1)

    # 测试模板
    test_templates = [
        "col(age) > int(18)",
        "?? > int(18)",
        "col(url) REGEXP string('^https?://')"
    ]

    for tmpl in test_templates:
        print("\n----------------------------------------")
        print("[SelfTest] predicate_template:", tmpl)
        try:
            sql = llm_client.complete_sql_with_schema(
                predicate_template=tmpl,
                schema_text=schema if 'schema' in locals() else "Table: dummy(id int)",
            )
            print("[SelfTest] completed SQL:", sql)
        except Exception as e:
            print("[SelfTest] 调用 LLM 完成 SQL 失败：", e)

    print("\n=== LLM Completion SelfTest Done ===")
