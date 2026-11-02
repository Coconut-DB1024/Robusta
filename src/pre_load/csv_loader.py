import csv
from typing import List, Optional
from .paths import get_data_path
import os


def load_examples_from_csv(filename: Optional[str] = None) -> List[str]:
    if filename is None:
        from RL.config import get_database_data_path

        csv_path = get_database_data_path("operators.csv")
    else:
        csv_path = os.path.join(get_data_path(), filename)

    examples: List[str] = []                   # 用于存放所有示例字符串

    # 打开 CSV 文件
    with open(csv_path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)             # 使用 DictReader 按列名读取

        # 遍历每一行记录
        for row in reader:
            # 依次读取 example_1 / example_2
            for key in (f"example_{i}" for i in range(1, 113)):
                raw_value = row.get(key, "")   # 安全获取列值
                if raw_value is None:          # 如果值为 None
                    continue                   # 跳过
                value = str(raw_value).strip() # 转为字符串并去掉首尾空白
                if not value:                  # 如果是空字符串
                    continue                   # 跳过
                examples.append(value)         # 加入示例列表

    return examples
