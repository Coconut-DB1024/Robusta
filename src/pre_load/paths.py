import os


def get_project_root() -> str:
    current_file = os.path.abspath(__file__)      # 当前文件 paths.py 的绝对路径
    pre_load_dir = os.path.dirname(current_file)  # 当前文件所在目录，即 pre_load 目录
    project_root = os.path.dirname(pre_load_dir)  # pre_load 的上一级目录，即项目根目录
    return project_root                           # 返回项目根目录路径


def get_data_path() -> str:
    root = get_project_root()                     # 获取项目根目录
    return os.path.join(root, "data")             # 拼接 data 目录路径并返回


def get_output_path() -> str:
    root = get_project_root()                     # 获取项目根目录
    return os.path.join(root, "output")