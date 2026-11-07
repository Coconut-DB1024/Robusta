# reappear-bugs

| No. | Database | Bug Title | Bug Link |
| --- | --- | --- | --- |
| 1 | SQLite | SQLite 3.47.0: incorrect REAL conversion for large numeric literal | [Link](https://sqlite.org/forum/info/5e3852a09127616b) |
| 2 | SQLite | SQLite 3.44.1: incorrect IS NULL result for empty-set aggregation with LEFT JOIN | [Link](https://sqlite.org/forum/info/5b4fc7d582bfa903) |
| 3 | SQLite | SQLite 3.53.0: incorrect EXISTS subquery result with unary plus operator and NULL index | [Link](https://sqlite.org/forum/info/f6fa0afcd2d1d98e) |
| 4 | 达梦数据库 | 疑似查询优化器问题：VIEW 经 UNION ALL 过滤后出现额外结果行 | [Link](https://eco.dameng.com/community/question/e170e240a1861b4633a136a04e9f0a40) |
| 5 | PostgreSQL | BUG #19709: Incorrect result when comparing OLD tuple column to itself in INSERT RETURNING | [Link](https://www.postgresql.org/message-id/19709-388df6818f7ef822%40postgresql.org) |
| 6 | PostgreSQL | BUG #19707: Reproduction of BUG #19553 on PostgreSQL 18.4: nested LEFT JOIN returns a constant instead of NULL | [Link](https://www.postgresql.org/message-id/19707-da66e5a04b617fd4%40postgresql.org) |
| 7 | DuckDB | Logic evaluation error with multi-column NOT IN when encountering NULL values | [Link](https://github.com/duckdb/duckdb/issues/22418) |
| 8 | Dolt | Incorrect NULL filtering with nullable projection and ORDER BY | [Link](https://github.com/dolthub/dolt/issues/11589) |
| 9 | MariaDB | Incorrect constant folding for NOT/IN predicate: scalar evaluation returns TRUE but WHERE returns no rows | [Link](https://jira.mariadb.org/browse/MDEV-41236) |
| 10 | MariaDB | XOR incorrectly treats a non-integer string operand as FALSE | [Link](https://jira.mariadb.org/browse/MDEV-41239) |
| 11 | DuckDB | Row-value equality with NULL returns TRUE instead of NULL | [Link](https://github.com/duckdb/duckdb/issues/22864) |
| 12 | Dolt | LIKE does not honor MySQL's default backslash escape | [Link](https://github.com/dolthub/dolt/issues/11910) |
| 13 | PostgreSQL | BUG #19721: json_value with DEFAULT ON ERROR returns inconsistent results for NULL input in materialized CTE | [Link](https://www.postgresql.org/message-id/19721-6d5c523ab7ee25e4%40postgresql.org) |
| 14 | MySQL | REPLACE function result truncated to original VARCHAR column length, causes wrong UNION deduplication | [Link](https://bugs.mysql.com/bug.php?id=121373) |
| 15 | PostgreSQL | BUG #19722: Window PARTITION BY numeric treats equal values with different scales as separate partitions | [Link](https://www.postgresql.org/message-id/19722-1b775eae8675c158%40postgresql.org) |
| 16 | MariaDB | MariaDB returns inconsistent results for a simple CASE predicate in WHERE when its WHEN operand contains double negation on a string column | [Link](https://jira.mariadb.org/browse/MDEV-41367) |
| 17 | MySQL | Possible incorrect error under derived condition pushdown with NULL and DATE predicates | [Link](https://bugs.mysql.com/121421) |
| 18 | MySQL | JSON Duality View INSERT silently drops a nested child row on secondary UNIQUE conflict | [Link](https://bugs.mysql.com/121426) |
| 19 | MySQL | MySQL Server crashes with an access violation when INSERT into a JSON Duality View repeats the root row as a nested sing | [Link](https://bugs.mysql.com/121427) |
| 20 | SQLite | SQLite 3.53.1: incorrect total() result after sliding-window frame transition | [Link](https://sqlite.org/forum/info/5fb137210df88ef5) |
| 21 | PostgreSQL | BUG #19751: MIN aggregate + FETCH FIRST 1 ROW WITH TIES causes false 21000 error when index scan is forced | [Link](https://www.postgresql.org/message-id/19751-db95b8241f0e33d9%40postgresql.org) |
