# 复现的bug

| 数据库 | Bug 名称 | Bug 链接 |
| --- | --- | --- |
| SQLite | SQLite 3.47.0: incorrect REAL conversion for large numeric literal | [链接](https://sqlite.org/forum/info/5e3852a09127616b) |
| SQLite | SQLite 3.44.1: incorrect IS NULL result for empty-set aggregation with LEFT JOIN | [链接](https://sqlite.org/forum/info/5b4fc7d582bfa903) |
| SQLite | SQLite 3.53.0: incorrect EXISTS subquery result with unary plus operator and NULL index | [链接](https://sqlite.org/forum/info/f6fa0afcd2d1d98e) |
| 达梦数据库 | 疑似查询优化器问题：VIEW 经 UNION ALL 过滤后出现额外结果行 | [链接](https://eco.dameng.com/community/question/e170e240a1861b4633a136a04e9f0a40) |
| PostgreSQL | BUG #19709: Incorrect result when comparing OLD tuple column to itself in INSERT RETURNING | [链接](https://www.postgresql.org/message-id/19709-388df6818f7ef822%40postgresql.org) |
| PostgreSQL | BUG #19707: Reproduction of BUG #19553 on PostgreSQL 18.4: nested LEFT JOIN returns a constant instead of NULL | [链接](https://www.postgresql.org/message-id/19707-da66e5a04b617fd4%40postgresql.org) |
| DuckDB | Logic evaluation error with multi-column NOT IN when encountering NULL values | [链接](https://github.com/duckdb/duckdb/issues/22418) |
| Dolt | Incorrect NULL filtering with nullable projection and ORDER BY | [链接](https://github.com/dolthub/dolt/issues/11589) |
| MariaDB | Incorrect constant folding for NOT/IN predicate: scalar evaluation returns TRUE but WHERE returns no rows | [链接](https://jira.mariadb.org/browse/MDEV-41236) |
| MariaDB | XOR incorrectly treats a non-integer string operand as FALSE | [链接](https://jira.mariadb.org/browse/MDEV-41239) |
| DuckDB | Row-value equality with NULL returns TRUE instead of NULL | [链接](https://github.com/duckdb/duckdb/issues/22864) |
| Dolt | LIKE does not honor MySQL's default backslash escape | [链接](https://github.com/dolthub/dolt/issues/11910) |
| PostgreSQL | BUG #19721: json_value with DEFAULT ON ERROR returns inconsistent results for NULL input in materialized CTE | [链接](https://www.postgresql.org/message-id/19721-6d5c523ab7ee25e4%40postgresql.org) |
| MySQL | REPLACE function result truncated to original VARCHAR column length, causes wrong UNION deduplication | [链接](https://bugs.mysql.com/bug.php?id=121373) |
| PostgreSQL | BUG #19722: Window PARTITION BY numeric treats equal values with different scales as separate partitions | [链接](https://www.postgresql.org/message-id/19722-1b775eae8675c158%40postgresql.org) |
| MariaDB | MariaDB returns inconsistent results for a simple CASE predicate in WHERE when its WHEN operand contains double negation on a string column | [链接](https://jira.mariadb.org/browse/MDEV-41367) |
| MySQL | Possible incorrect error under derived condition pushdown with NULL and DATE predicates | [链接](https://bugs.mysql.com/121421) |
| MySQL | JSON Duality View INSERT silently drops a nested child row on secondary UNIQUE conflict | [链接](https://bugs.mysql.com/121426) |
| MySQL | MySQL Server crashes with an access violation when INSERT into a JSON Duality View repeats the root row as a nested sing | [链接](https://bugs.mysql.com/121427) |
| SQLite | SQLite 3.53.1: incorrect total() result after sliding-window frame transition | [链接](https://sqlite.org/forum/info/5fb137210df88ef5) |
| PostgreSQL | BUG #19751: MIN aggregate + FETCH FIRST 1 ROW WITH TIES causes false 21000 error when index scan is forced | [链接](https://www.postgresql.org/message-id/19751-db95b8241f0e33d9%40postgresql.org) |
