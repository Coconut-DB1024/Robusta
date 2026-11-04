The root `main.py` exports two result files here after a successful experiment:

- `satisfiable_predicates.txt`: the satisfiable TLP variants, with the original heading and without the two rate lines.
- `experiment_summary.txt`: final RL graph node/edge counts, executability and satisfiability rates, mean similarity, population standard deviation, and predicate generation speeds for similarity <=40% and <=80%.

Generation speed uses the generated predicate count divided by traversal/rendering time plus TLP test-SQL construction time, excluding database validation. Each threshold speed multiplies that speed by the corresponding fraction of scored predicates, including equality at the threshold. Rates use two decimal places; similarity statistics and speeds use four.

Old outputs are removed before the new exports are installed. This README is preserved. Original outputs remain in `src/`; missing current Stage-3 reports leave existing exports intact, and unavailable statistics are shown as `N/A`.
