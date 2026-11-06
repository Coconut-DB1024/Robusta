import os
from typing import List
from .csv_loader import load_examples_from_csv
from .predicate_parser import parse_predicate_example
from .paths import get_output_path


def pre_load() -> List[str]: #预处理入口
    from RL.config import get_database_data_path

    # 读取所有示例
    examples = load_examples_from_csv()
    get_database_data_path("Incorrect_Collection.txt").read_text(encoding="utf-8")

    # 解析成模板
    all_templates: List[str] = []
    for ex in examples:
        tpl = parse_predicate_example(ex)   # 调用解析函数
        all_templates.append(tpl)           # 收集模板

    # 去重：保持顺序的同时去掉重复
    unique_templates: List[str] = []        # 存放去重后的模板
    seen = set()                            # 用于记录已出现过的模板字符串
    for tpl in all_templates:
        if tpl in seen:                     # 如果已经出现过
            continue                        # 跳过重复
        seen.add(tpl)                       # 标记为已出现
        unique_templates.append(tpl)        # 追加到结果列表

    # 写入 output/predicate.txt
    output_dir = get_output_path()          # 获取 output 目录
    os.makedirs(output_dir, exist_ok=True)  # 不存在则创建目录

    output_file = os.path.join(output_dir, "predicate.txt")  # 输出文件路径
    with open(output_file, mode="w", encoding="utf-8") as f:
        for line in unique_templates:       # 遍历去重后的模板
            f.write(line + "\n")           # 每个模板一行

    from .llm_seed import supplement_predicates

    return supplement_predicates(output_file, unique_templates)
