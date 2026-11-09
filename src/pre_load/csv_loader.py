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

    examples: List[str] = []                   

    with open(csv_path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)             

        for row in reader:
            for key in (f"example_{i}" for i in range(1, 113)):
                raw_value = row.get(key, "")   
                if raw_value is None:          
                    continue                   
                value = str(raw_value).strip() 
                if not value:                  
                    continue                   
                examples.append(value)         

    return examples
