# Robusta

- `data/`: Stores experiment results archived by the root launcher.
- `exp/`: Intended for scripts used to launch the project.
- `src/`: Contains the project source code and supporting resources.

Currently, configure the database connection in `Database_link.json`, then run `python main.py` from the `Robusta` root directory to start the complete experiment. After the run finishes successfully, its results are archived in a timestamped directory under `Robusta/data/`: `global_knowledge_graph.json` contains the final RL knowledge graph; `satisfiable_predicates.txt` contains the Stage-3 executability rate, satisfiability rate, and satisfiable TLP predicates; and `similarity/` contains the similarity analysis results and the matching text summary of mean similarity and standard deviation. The original outputs remain in `src/`. If an output was not generated during the current run, the launcher reports it instead of copying an older result.

Run `download_codet5_small.py` to download the required [CodeT5-small model](https://huggingface.co/Salesforce/codet5-small) from Hugging Face into `src/codet5-small/`. From the `Robusta` directory, install the dependencies if needed, then run:

```bash
python -m pip install "torch>=2.6" "transformers==4.57.1" "tokenizers==0.22.1" "safetensors==0.6.2"
python download_codet5_small.py
```

The script uses a fixed model revision and verifies SHA-256 checksums to reproduce the project's original eight model files, including `model.safetensors`. The download location is relative to the script, regardless of the working directory. Existing matching files are kept, and temporary download files are cleaned up automatically.
