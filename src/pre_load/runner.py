import os
from typing import List
from .csv_loader import load_examples_from_csv
from .predicate_parser import parse_predicate_example
from .paths import get_output_path


def pre_load() -> List[str]: 
    from RL.config import get_database_data_path

    examples = load_examples_from_csv()
    get_database_data_path("Incorrect_Collection.txt").read_text(encoding="utf-8")

    all_templates: List[str] = []
    for ex in examples:
        tpl = parse_predicate_example(ex)   
        all_templates.append(tpl)           

    unique_templates: List[str] = []        
    seen = set()                            
    for tpl in all_templates:
        if tpl in seen:                     
            continue                        
        seen.add(tpl)                       
        unique_templates.append(tpl)        

    output_dir = get_output_path()          
    os.makedirs(output_dir, exist_ok=True)  

    output_file = os.path.join(output_dir, "predicate.txt")  
    with open(output_file, mode="w", encoding="utf-8") as f:
        for line in unique_templates:       
            f.write(line + "\n")           

    from .llm_seed import supplement_predicates

    return supplement_predicates(output_file, unique_templates)
