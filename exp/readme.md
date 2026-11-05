This directory stores experiment launch scripts.

`full_experiment/main.py` launches the complete experiment using the existing project configuration. From the `Robusta` root directory, run:

```bash
python exp/full_experiment/main.py
```

`llm/main.py` launches the independent LLM experiment using the PostgreSQL and `online_model` settings in `Robusta/configuration.json`. From the `Robusta` root directory, run:

```bash
python exp/llm/main.py
```
