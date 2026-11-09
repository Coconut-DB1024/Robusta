import os
import numpy as np


EMB_FILE = "output/predicate_embeddings_codet5.npy"


PREDICATE_FILE = "output/predicate.txt"


def main():
    if not os.path.exists(EMB_FILE):
        raise FileNotFoundError(f'Embedding file not found: {EMB_FILE}')

    embeddings = np.load(EMB_FILE)  

    print('[INFO] Embedding matrix loaded successfully.')
    print(f'[INFO] Embedding matrix shape: {embeddings.shape}')  
    print(f'[INFO] Embedding data type: {embeddings.dtype}')

    predicates = []
    if os.path.exists(PREDICATE_FILE):
        with open(PREDICATE_FILE, "r", encoding="utf-8") as f:
            predicates = [line.strip() for line in f if line.strip()]
        print(f'[INFO] Predicate text lines: {len(predicates)}')
    else:
        print(f'[Warning] Predicate text file not found: {PREDICATE_FILE}; displaying embeddings only.')

    num_to_show = min(10, embeddings.shape[0])
    for i in range(num_to_show):
        print("\n==============================")
        print(f'[DEBUG] Embedding for predicate {i}:')
        if i < len(predicates):
            print(f'  Original predicate: {predicates[i]}')
        print(f'  Embedding shape: {embeddings[i].shape}')  
        print(f'  First 10 dimensions: {embeddings[i][:20]}')  
    print('\n[INFO] read_predicate_embeddings.py completed.')


if __name__ == "__main__":
    main()