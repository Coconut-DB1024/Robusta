"""
文件名: db_initializer.py
路径: BugFinding/db_initializer.py
描述:
    数据库环境初始化脚本。
    该脚本负责创建一个包含当前 DBMS 支持的所有数据类型的"大宽表" (Wide Table)，
    并填充带有大量 NULL 值和边界值的随机数据，为后续的 Fuzzing 测试提供靶场。

    依赖:
    - RL.config: 获取数据库配置信息
    - RL.db_executor: 执行 SQL 语句
"""

import os
import json
import random
import string
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from pathlib import Path

# ================= 导入项目底层依赖 =================
import sys
# 获取当前文件的父目录的父目录，即项目根目录
project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将项目根目录添加到系统路径，以便可以导入 RL 模块
sys.path.append(project_root_path)

# 复用你提供的配置模块
from RL.config import PROJECT_ROOT, DB_CONFIG, get_target_table
# 复用你提供的数据库执行器模块
from RL.db_executor import MySQLExecutor

# ================= 全局配置常量 =================
# 类型配置文件的存储目录
TYPE_CONFIG_DIR = PROJECT_ROOT / "BugFinding" / "db_types"
# 要创建的测试用"大宽表"的名称
TABLE_NAME, _, TABLE_SQL = get_target_table("mysql", DB_CONFIG)
# 初始化时需要插入的数据行数
INITIAL_ROWS = 500
# 随机生成 NULL 值的概率 (0.15 代表 15% 的概率生成 NULL)
# 这是 TLP (三值逻辑) 测试能发现 Bug 的关键参数
NULL_PROBABILITY = 0.15
# 批量插入时的批次大小，防止单条 SQL 语句过长导致报错
BATCH_SIZE = 200

# 配置日志输出格式，方便调试
logging.basicConfig(level=logging.INFO, format='[Initializer] %(message)s')
logger = logging.getLogger(__name__)


class DataTypeConfigLoader:
    """
    类说明: 负责加载数据类型配置文件。
    它会根据 config 中配置的 db_type (如 mysql) 自动查找对应的 JSON 文件。
    """
    def __init__(self):
        # 用于存储加载后的类型映射字典
        self.type_map = {}
        # 初始化时立即执行加载逻辑
        self._load_config()

    def _load_config(self):
        """
        加载配置文件的核心逻辑。
        修改说明: 现在加载的文件名后缀统一为 _types.json
        """
        # 从全局配置中获取当前数据库类型 (例如 "mysql")
        db_type = DB_CONFIG.db_type

        # 构造特定数据库的配置文件路径 (例如 db_types/mysql_types.json)
        # 这里进行了修改，增加了 _types 后缀，防止文件名混淆
        specific_config = TYPE_CONFIG_DIR / f"{db_type}_types.json"

        # 构造通用的配置文件路径 (例如 db_types/common_types.json)
        # 这里同样进行了修改
        common_config = TYPE_CONFIG_DIR / "common_types.json"

        target_file = None
        # 优先查找特定数据库的配置
        if os.path.exists(specific_config):
            logger.info(f"检测到特定数据库配置，正在加载: {specific_config}")
            target_file = specific_config
        # 如果找不到，降级使用通用配置
        elif os.path.exists(common_config):
            logger.info(f"未找到 {db_type}_types.json，将加载通用配置: {common_config}")
            target_file = common_config
        else:
            # 如果两个都找不到，抛出异常，因为无法继续初始化
            raise FileNotFoundError(f"致命错误: 在 {TYPE_CONFIG_DIR} 下未找到任何可用的类型配置文件！")

        # 打开并解析 JSON 文件
        with open(target_file, 'r', encoding='utf-8') as f:
            self.type_map = json.load(f)

    def get_all_types(self) -> Dict[str, List[str]]:
        """
        返回加载的数据类型字典。
        格式示例: {'INTEGER': ['INT', 'BIGINT'], 'STRING': ['TEXT', 'CHAR']}
        """
        return self.type_map


class RandomDataFactory:
    """
    类说明: 随机数据工厂。
    根据传入的 'Category' (大类)，生成符合该类型的随机数据 (Python对象或SQL字符串)。
    """

    @staticmethod
    def generate_value(category: str) -> Any:
        """
        核心方法: 根据类型分类分发生成逻辑。
        参数 category: 类型的大类，如 'INTEGER', 'STRING' (对应 JSON 的 Key)
        """
        # 1. 随机注入 NULL 值 (TLP测试的核心逻辑)
        if random.random() < NULL_PROBABILITY:
            return "NULL"

        # 2. 将类别转为大写，确保匹配无误
        category = category.upper()

        # 3. 根据类别调用对应的生成函数
        if "INTEGER" in category or "BOOLEAN" in category:
            return RandomDataFactory._gen_int()
        elif "FLOAT" in category:
            return RandomDataFactory._gen_float()
        elif "STRING" in category:
            return RandomDataFactory._gen_string()
        elif "DATETIME" in category:
            return RandomDataFactory._gen_datetime()
        elif "BINARY" in category:
            return RandomDataFactory._gen_binary()
        elif "JSON" in category:
            return RandomDataFactory._gen_json()
        else:
            # 如果遇到未知的类型，默认作为字符串处理，保证代码不崩溃
            return RandomDataFactory._gen_string()

    # --- 以下是内部具体的生成实现 ---

    @staticmethod
    def _gen_int():
        # 生成一个随机整数，范围覆盖正负数，用于测试边界
        return random.randint(-10000, 10000)

    @staticmethod
    def _gen_float():
        # 生成一个随机浮点数，保留两位小数
        return round(random.uniform(-10000.0, 10000.0), 2)

    @staticmethod
    def _gen_string():
        # 随机决定字符串长度 (1 到 50 个字符)
        length = random.randint(1, 50)
        # 字符集包含字母、数字、空格和部分特殊符号
        chars = string.ascii_letters + string.digits + " _-@"
        # 生成字符串
        val = ''.join(random.choice(chars) for _ in range(length))
        # 返回带单引号的 SQL 字符串格式
        return f"'{val}'"

    @staticmethod
    def _gen_datetime():
        # 生成一个随机时间点，范围在过去 10 年到当前时间
        start = datetime.now() - timedelta(days=365*10)
        end = datetime.now()
        # 计算随机时间戳
        random_date = start + (end - start) * random.random()
        # 格式化为 SQL 标准日期格式
        return f"'{random_date.strftime('%Y-%m-%d %H:%M:%S')}'"

    @staticmethod
    def _gen_binary():
        # 模拟二进制数据，为了兼容性，这里生成一个十六进制样式的字符串
        # 例如: 'abc12345'，在 SQL 中通常可以直接存入 BLOB 或以字符串形式处理
        random_hex = ''.join(random.choices('0123456789ABCDEF', k=16))
        return f"'{random_hex}'"

    @staticmethod
    def _gen_json():
        # 生成一个简单的 JSON 对象
        data = {
            "id": random.randint(1, 100),
            "tag": "".join(random.choices(string.ascii_lowercase, k=5)),
            "score": round(random.random(), 2)
        }
        # 将 Python 字典转换为 JSON 字符串
        json_str = json.dumps(data)
        # 注意：在 SQL 中插入 JSON 字符串需要单引号包裹
        return f"'{json_str}'"


class DatabaseInitializer:
    """
    类说明: 数据库初始化主控类。
    负责协调配置加载、表结构创建和数据填充的完整流程。
    """
    def __init__(self):
        # 实例化配置加载器
        self.config_loader = DataTypeConfigLoader()

        # 实例化数据库执行器 (复用 RL.db_executor)
        # autoconnect=True 表示初始化时自动建立连接
        self.executor = MySQLExecutor(autoconnect=True, config=DB_CONFIG)

        # 用于在内存中临时存储表结构元数据
        # 列表元素结构: (列名, SQL完整类型定义, 生成器分类)
        self.columns_meta = []

    def setup_database(self):
        """
        主控方法: 执行初始化的全过程。
        """
        try:
            logger.info(f"=== 开始初始化数据库实例 (目标表: {TABLE_NAME}) ===")

            # 第一步: 准备元数据 (决定要建哪些列)
            self._prepare_columns_metadata()

            # 第二步: 重建表结构 (先删除旧表，再建新表)
            self._recreate_table()

            # 第三步: 填充随机数据 (批量插入)
            self._populate_data()

            logger.info("=== 数据库初始化任务圆满完成 ===")

        except Exception as e:
            # 捕获所有异常并记录日志，方便排查
            logger.error(f"初始化过程中发生严重错误: {e}")
            # 继续向上抛出异常，通知上层调用者失败了
            raise
        finally:
            # 无论成功还是失败，都要确保关闭数据库连接
            if self.executor:
                self.executor.close()
                logger.info("数据库连接已关闭。")

    def _prepare_columns_metadata(self):
        """
        方法说明: 解析配置，构建列的元数据列表。
        策略: 遍历 JSON 中定义的每一个具体类型，都为它创建一个专属列。
        这样可以确保"大宽表"包含数据库支持的所有类型。
        """
        # 获取类型字典
        type_map = self.config_loader.get_all_types()
        self.columns_meta = []

        # 总是包含一个自增主键 id，方便管理
        # 注意: PRIMARY_KEY_AUTO 是我们内部的一个标记，非 SQL 类型
        self.columns_meta.append(('id', 'PRIMARY_KEY_AUTO', 'ID'))

        # 遍历配置中的大类 (如 INTEGER)
        for category, type_list in type_map.items():
            # 遍历大类下的具体类型 (如 TINYINT, BIGINT)
            for idx, sql_type in enumerate(type_list):
                # 构造列名: col_类型类别_序号 (例如 col_integer_0)
                # 使用 lower() 确保列名小写，符合习惯
                safe_category = category.lower()
                col_name = f"col_{safe_category}_{idx}"

                # 将 (列名, SQL类型, 类别) 存入列表
                # sql_type 是直接用于 CREATE TABLE 的 (如 VARCHAR(255))
                # category 用于后续告诉 RandomDataFactory 应该生成什么样的数据
                self.columns_meta.append((col_name, sql_type, category))

        logger.info(f"元数据准备完毕，计划创建 {len(self.columns_meta)} 个列。")

    def _recreate_table(self):
        """
        方法说明: 执行 DROP 和 CREATE TABLE 操作。
        """
        # 1. 构建并执行 DROP TABLE 语句
        drop_sql = f"DROP TABLE IF EXISTS {TABLE_SQL};"
        self.executor.execute_query(drop_sql)

        # 2. 构建 CREATE TABLE 语句
        col_defs = []

        # 遍历元数据，拼接列定义
        for col_name, sql_type, _ in self.columns_meta:
            if col_name == 'id':
                # 对于主键 id，使用 MySQL 特有的自增语法
                # 如果是其他数据库，这里可能需要调整适配，但目前主要针对 MySQL
                col_defs.append("id INT AUTO_INCREMENT PRIMARY KEY")
            else:
                # 普通列: 列名 + 类型 (例如: col_integer_0 TINYINT)
                col_defs.append(f"{col_name} {sql_type}")

        # 拼接完整的 CREATE 语句
        create_sql = f"CREATE TABLE {TABLE_SQL} ({', '.join(col_defs)});"

        logger.info(f"正在执行建表语句...")
        # 调用执行器
        res = self.executor.execute_query(create_sql)

        # 检查执行结果
        if not res.success:
            raise Exception(f"建表失败，数据库返回错误: {res.error}")
        logger.info("表结构创建成功。")

    def _populate_data(self):
        """
        方法说明: 生成随机数据并执行批量插入。
        """
        logger.info(f"开始生成并插入数据，目标行数: {INITIAL_ROWS}，NULL概率: {NULL_PROBABILITY}")

        # 获取所有需要插入数据的列 (排除 id，因为它由数据库自动生成)
        target_cols = [meta for meta in self.columns_meta if meta[0] != 'id']

        # 提取列名部分，用于组装 INSERT INTO table (col1, col2...)
        col_names_str = ", ".join([c[0] for c in target_cols])

        # 用于缓存一批数据的列表
        batch_values = []
        # 记录总插入数
        total_inserted = 0

        # 循环生成数据
        for i in range(INITIAL_ROWS):
            row_data = []
            # 遍历每一列，生成对应的数据
            for col_name, sql_type, category in target_cols:
                # 调用工厂生成值
                val = RandomDataFactory.generate_value(category)
                # 必须转换为字符串才能拼接到 SQL 中
                row_data.append(str(val))

            # 将一行数据拼装成 (v1, v2, v3) 的格式
            row_str = f"({', '.join(row_data)})"
            batch_values.append(row_str)

            # 检查是否达到批量插入的阈值
            if len(batch_values) >= BATCH_SIZE:
                # 执行插入
                self._insert_batch(col_names_str, batch_values)
                # 更新计数器
                total_inserted += len(batch_values)
                # 清空缓存
                batch_values = []
                # 打印进度 (不换行)
                print(f"\r[Progress] 已插入: {total_inserted}/{INITIAL_ROWS} 行...", end="")

        # 循环结束后，如果缓存里还有剩余数据，一次性插入
        if batch_values:
            self._insert_batch(col_names_str, batch_values)
            total_inserted += len(batch_values)

        print("") # 打印换行
        logger.info(f"数据填充完成，共插入 {total_inserted} 行。")

    def _insert_batch(self, col_names: str, values_list: List[str]):
        """
        方法说明: 执行单次批量 INSERT 操作。
        参数 col_names: 列名字符串 "col1, col2"
        参数 values_list: 数据行列表 ["(1, 'a')", "(2, 'b')"]
        """
        if not values_list:
            return

        # 构造 SQL: INSERT INTO table (cols) VALUES (row1), (row2)...
        sql = f"INSERT INTO {TABLE_SQL} ({col_names}) VALUES {', '.join(values_list)};"

        # 执行 SQL
        res = self.executor.execute_query(sql)

        if not res.success:
            # 批量插入失败通常是严重的 (如数据类型不匹配)，这里选择抛出异常中断测试
            # 如果希望容错，可以将 raise 改为 logger.error
            logger.error(f"批量插入发生错误: {res.error}")
            raise Exception(f"数据插入失败: {res.error}")


if __name__ == "__main__":
    # 该脚本既可以作为模块被 import，也可以直接运行用于测试环境搭建
    initializer = DatabaseInitializer()
    initializer.setup_database()
