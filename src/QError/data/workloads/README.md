# 评估工作负载

`job-light.csv`、`job-light.bitmaps` 和 `job-light.sql` 是原来的 70 条 JOB-light 评估数据。

`job-exclude-evaluated.*` 和对应的索引 CSV 保留旧包中的 42 条 JOB-exclude 评估数据，每个查询编号占一行。`job-exclude/` 是历史复现入口使用的输入，按原流程对去重 SQL 推理，再映射回查询编号并排除零基数查询 8b。两种存放方式下，最终 42 条查询的编号、SQL 哈希、真实基数及 bitmap 一致。

