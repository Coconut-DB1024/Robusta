This directory contains scripts for running the experiments in Robusta. All commands should be executed from the Robusta root directory.

**Database Initialization**

Initialize the database table using the database connection settings and `table_name` in `configuration.json`:

```bash
python exp/db_initializer/db_initializer.py
```

The selected database must already exist. Each run drops the existing table with the configured name, recreates it, and populates it with test data.

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

The experimental results will be saved in the `data/LLM_reuslts/` directory.

**L-error Experiment**

Run the independent L-error experiment using the PostgreSQL settings and experiment selection in `configuration-lerror.json`:

```bash
python exp/L-error/main.py
```

Only PostgreSQL is supported. Each selected experiment archives `results.csv` and `l_error_summary.txt` in `data/L-error_results/<experiment>_results/`.

**Q-error Experiment**

Run the independent Q-error experiment using the settings and experiment selection in `configuration-qerror.json`:

```bash
python exp/Qerror/main.py
```

The experimental results will be saved in the `data/Qerror/<experiment>/` directory.

