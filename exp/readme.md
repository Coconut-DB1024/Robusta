This directory contains scripts for running the experiments in Robusta. All commands should be executed from the Robusta root directory.

**Full Experiment**

Run the complete end-to-end workflow using the settings in `configuration.json`:

```bash
python exp/full_experiment/main.py
```

The experimental results will be saved in the `data/` directory.

**LLM Experiment**

Run the independent LLM experiment using the PostgreSQL and `online_model` settings in `configuration.json`:

```bash
python exp/llm/main.py
```

The experimental results will be saved in the `data/LLM_reuslts/` directory.
