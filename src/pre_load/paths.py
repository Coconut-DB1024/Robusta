import os


def get_project_root() -> str:
    current_file = os.path.abspath(__file__)      
    pre_load_dir = os.path.dirname(current_file)  
    project_root = os.path.dirname(pre_load_dir)  
    return project_root                           


def get_data_path() -> str:
    root = get_project_root()                     
    return os.path.join(root, "data")             


def get_output_path() -> str:
    root = get_project_root()                     
    return os.path.join(root, "output")