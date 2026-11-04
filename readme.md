# Robusta

- `data/`: Stores experiment results archived by the root launcher.
- `exp/`: Intended for scripts used to launch the project.
- `src/`: Contains the project source code and supporting resources.

Currently, configure the database connection in `Database_link.json`, then run `python main.py` from the `Robusta` root directory to start the complete experiment. After a successful run, `Robusta/data/` contains two result files: `satisfiable_predicates.txt` contains the satisfiable TLP variants without the two rate lines, and `experiment_summary.txt` contains the final RL graph node/edge counts, executability and satisfiability rates, mean similarity, standard deviation, and predicate generation speeds for similarity <=40% and <=80%. Previous outputs in `data/` are replaced; `data/readme.md` and the original outputs in `src/` are retained. If current Stage-3 outputs are missing, the launcher keeps the previous results. Console output is flushed promptly; redirect it to a file if a console log is needed.

The base generation speed is the Stage-3 generated predicate count divided by predicate traversal/rendering time plus TLP test-SQL construction time. Database validation, graph preparation, and similarity analysis are excluded. Each reported speed multiplies the base speed by the fraction of scored predicates with similarity less than or equal to its threshold; unscored predicates are excluded from that fraction. Satisfiability is measured among executable predicates. Rates use two decimal places, and similarity statistics and speeds use four; unavailable statistics are shown as `N/A`.

Run `download_codet5_small.py` to download the required [CodeT5-small model](https://huggingface.co/Salesforce/codet5-small) from Hugging Face into `src/codet5-small/`. From the `Robusta` directory, install the dependencies if needed, then run:

```bash
python -m pip install "torch>=2.6" "transformers==4.57.1" "tokenizers==0.22.1" "safetensors==0.6.2"
python download_codet5_small.py
```

The script uses a fixed model revision and verifies SHA-256 checksums to reproduce the project's original eight model files, including `model.safetensors`. The download location is relative to the script, regardless of the working directory. Existing matching files are kept, and temporary download files are cleaned up automatically.
