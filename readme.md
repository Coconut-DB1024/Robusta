# Robusta

- `data/`: Stores experiment results archived by the root launcher.
- `exp/`: Intended for scripts used to launch the project.
- `src/`: Contains the project source code and supporting resources.

Currently, configure the database connection in `configuration.json`, then run `python main.py` from the `Robusta` root directory to start the complete experiment. After a successful run, `Robusta/data/` contains two result files: `satisfiable_predicates.txt` contains the satisfiable TLP variants without the two rate lines, and `experiment_summary.txt` contains the final RL graph node/edge counts, executability and satisfiability rates, mean similarity, standard deviation, and predicate generation speeds for similarity <=40% and <=80%. Previous outputs in `data/` are replaced; If current outputs are missing, the launcher keeps the previous results. 

First, install the project dependencies from `requirements.txt`. From the `Robusta` directory, run:

```bash
python -m pip install -r requirements.txt
```

Then run `download_codet5_small.py` to download the required [CodeT5-small model](https://huggingface.co/Salesforce/codet5-small) from Hugging Face into `src/codet5-small/`:

```bash
python download_codet5_small.py
```

Alternatively, you can download codet5-small.zip from GitHub Releases and extract it into the root directory of the Robusta project.
