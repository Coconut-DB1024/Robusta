This directory contains scripts for running the experiments in Robusta. All commands should be executed from the Robusta root directory.

**Robusta Experiment**

Run the complete end-to-end workflow using the settings in `configuration.json`:

```bash
python exp/Robusta_experiment/main.py
```

The experimental results will be saved in the `data/` directory.

**LLM Experiment**

Run the independent LLM experiment using the PostgreSQL and `online_model` settings in `configuration.json`:

```bash
python exp/llm/main.py
```

Only PostgreSQL is supported. The experimental results will be saved in the `data/LLM_reuslts/` directory.

**L-error Experiment**

Run the independent L-error experiment using the PostgreSQL settings and experiment selection in `configuration-lerror.json`:

```bash
python exp/L-error/main.py
```

Only PostgreSQL is supported. The experimental results will be saved in the `data/L-error_results/<experiment>_results/`.
