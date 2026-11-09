import os
import json
import random
import string
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from pathlib import Path


import sys

project_root_path = str(Path(__file__).resolve().parents[2] / "src")

if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)


from RL.config import PROJECT_ROOT, DB_CONFIG, get_target_table

from RL.db_executor import MySQLExecutor


TYPE_CONFIG_DIR = PROJECT_ROOT / "BugFinding" / "db_types"

TABLE_NAME, _, TABLE_SQL = get_target_table("mysql", DB_CONFIG)

INITIAL_ROWS = 500


NULL_PROBABILITY = 0.15

BATCH_SIZE = 200


logging.basicConfig(level=logging.INFO, format='[Initializer] %(message)s')
logger = logging.getLogger(__name__)


class DataTypeConfigLoader:
    def __init__(self):
        self.type_map = {}

        self._load_config()

    def _load_config(self):
        db_type = DB_CONFIG.db_type

        specific_config = TYPE_CONFIG_DIR / f"{db_type}_types.json"

        common_config = TYPE_CONFIG_DIR / "common_types.json"

        target_file = None

        if os.path.exists(specific_config):
            logger.info(f"检测到特定数据库配置，正在加载: {specific_config}")
            target_file = specific_config

        elif os.path.exists(common_config):
            logger.info(f"未找到 {db_type}_types.json，将加载通用配置: {common_config}")
            target_file = common_config
        else:
            raise FileNotFoundError(f"致命错误: 在 {TYPE_CONFIG_DIR} 下未找到任何可用的类型配置文件！")

        with open(target_file, 'r', encoding='utf-8') as f:
            self.type_map = json.load(f)

    def get_all_types(self) -> Dict[str, List[str]]:
        return self.type_map


class RandomDataFactory:
    @staticmethod
    def generate_value(category: str) -> Any:
        if random.random() < NULL_PROBABILITY:
            return "NULL"

        category = category.upper()

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
            return RandomDataFactory._gen_string()

    @staticmethod
    def _gen_int():
        return random.randint(-10000, 10000)

    @staticmethod
    def _gen_float():
        return round(random.uniform(-10000.0, 10000.0), 2)

    @staticmethod
    def _gen_string():
        length = random.randint(1, 50)

        chars = string.ascii_letters + string.digits + " _-@"

        val = ''.join(random.choice(chars) for _ in range(length))

        return f"'{val}'"

    @staticmethod
    def _gen_datetime():
        start = datetime.now() - timedelta(days=365*10)
        end = datetime.now()

        random_date = start + (end - start) * random.random()

        return f"'{random_date.strftime('%Y-%m-%d %H:%M:%S')}'"

    @staticmethod
    def _gen_binary():
        random_hex = ''.join(random.choices('0123456789ABCDEF', k=16))
        return f"'{random_hex}'"

    @staticmethod
    def _gen_json():
        data = {
            "id": random.randint(1, 100),
            "tag": "".join(random.choices(string.ascii_lowercase, k=5)),
            "score": round(random.random(), 2)
        }

        json_str = json.dumps(data)

        return f"'{json_str}'"


class DatabaseInitializer:
    def __init__(self):
        self.config_loader = DataTypeConfigLoader()

        self.executor = MySQLExecutor(autoconnect=True, config=DB_CONFIG)

        self.columns_meta = []

    def setup_database(self):
        try:
            logger.info(f"=== 开始初始化数据库实例 (目标表: {TABLE_NAME}) ===")

            self._prepare_columns_metadata()

            self._recreate_table()

            self._populate_data()

            logger.info("=== 数据库初始化任务圆满完成 ===")

        except Exception as e:
            logger.error(f"初始化过程中发生严重错误: {e}")

            raise
        finally:
            if self.executor:
                self.executor.close()
                logger.info("数据库连接已关闭。")

    def _prepare_columns_metadata(self):
        type_map = self.config_loader.get_all_types()
        self.columns_meta = []

        self.columns_meta.append(('id', 'PRIMARY_KEY_AUTO', 'ID'))

        for category, type_list in type_map.items():
            for idx, sql_type in enumerate(type_list):
                safe_category = category.lower()
                col_name = f"col_{safe_category}_{idx}"

                self.columns_meta.append((col_name, sql_type, category))

        logger.info(f"元数据准备完毕，计划创建 {len(self.columns_meta)} 个列。")

    def _recreate_table(self):
        drop_sql = f"DROP TABLE IF EXISTS {TABLE_SQL};"
        drop_result = self.executor.execute_query(drop_sql)
        if not drop_result.success:
            raise RuntimeError(f"Failed to drop target table: {drop_result.error}")

        col_defs = []

        for col_name, sql_type, _ in self.columns_meta:
            if col_name == 'id':
                col_defs.append("id INT AUTO_INCREMENT PRIMARY KEY")
            else:
                col_defs.append(f"{col_name} {sql_type}")

        create_sql = f"CREATE TABLE {TABLE_SQL} ({', '.join(col_defs)});"

        logger.info(f"正在执行建表语句...")

        res = self.executor.execute_query(create_sql)

        if not res.success:
            raise Exception(f"建表失败，数据库返回错误: {res.error}")
        logger.info("表结构创建成功。")

    def _populate_data(self):
        logger.info(f"开始生成并插入数据，目标行数: {INITIAL_ROWS}，NULL概率: {NULL_PROBABILITY}")

        target_cols = [meta for meta in self.columns_meta if meta[0] != 'id']

        col_names_str = ", ".join([c[0] for c in target_cols])

        batch_values = []

        total_inserted = 0

        for i in range(INITIAL_ROWS):
            row_data = []

            for col_name, sql_type, category in target_cols:
                val = RandomDataFactory.generate_value(category)

                row_data.append(str(val))

            row_str = f"({', '.join(row_data)})"
            batch_values.append(row_str)

            if len(batch_values) >= BATCH_SIZE:
                self._insert_batch(col_names_str, batch_values)

                total_inserted += len(batch_values)

                batch_values = []

                print(f"\r[Progress] 已插入: {total_inserted}/{INITIAL_ROWS} 行...", end="")

        if batch_values:
            self._insert_batch(col_names_str, batch_values)
            total_inserted += len(batch_values)

        print("")
        logger.info(f"数据填充完成，共插入 {total_inserted} 行。")

    def _insert_batch(self, col_names: str, values_list: List[str]):
        if not values_list:
            return

        sql = f"INSERT INTO {TABLE_SQL} ({col_names}) VALUES {', '.join(values_list)};"

        res = self.executor.execute_query(sql)

        if not res.success:
            logger.error(f"批量插入发生错误: {res.error}")
            raise Exception(f"数据插入失败: {res.error}")


if __name__ == "__main__":
    initializer = DatabaseInitializer()
    initializer.setup_database()
