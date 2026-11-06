# 独立发现的bug

| 数据库 | Bug 名称 | Bug 链接 |
| --- | --- | --- |
| MariaDB | IS JSON WITH UNIQUE KEYS accepts duplicate member names written with equivalent Unicode escapes | [链接](https://jira.mariadb.org/browse/MDEV-40635) |
| TiDB | TiDB returns incorrect JOIN results when explicit COLLATE is applied to a string expression | [链接](https://github.com/pingcap/tidb/issues/70860) |
| Dolt | IN predicate ignores utf8mb4_unicode_ci for accent-insensitive comparisons | [链接](https://github.com/dolthub/dolt/issues/11907) |
| Dolt | Incorrect CAST from BIGINT UNSIGNED to SIGNED | [链接](https://github.com/dolthub/dolt/issues/11906) |
| MariaDB | CREATE VIEW changes the result of an expression combining NOT, LIKE, and IN | [链接](https://jira.mariadb.org/browse/MDEV-41371) |
| TiDB | Partial index documentation lists IN predicates as supported, but TiDB v8.5.8 rejects them | [链接](https://github.com/pingcap/tidb/issues/71733) |
| MySQL | Bug #121435 derived_condition_pushdown causes incorrect REGEXP matching with JSON_TABLE under NO_BACKSLASH_ESCAPES mode | [链接](https://bugs.mysql.com/bug.php?id=121435) |
| MySQL | ST_Within returns different results when a geographic SPATIAL index is forced | [链接](https://bugs.mysql.com/bug.php?id=121475) |
| MySQL | VECTOR_TO_STRING() round-trip changes VECTOR value | [链接](https://bugs.mysql.com/bug.php?id=121516) |
| SQLite | Date modifiers return an invalid day after a 24:00 time value | [链接](https://sqlite.org/forum/info/c7ab8a11c8e3bd5b) |
| MariaDB | VEC_DISTANCE_COSINE returns 0 for small-magnitude non-zero orthogonal and opposite vectors | [链接](https://jira.mariadb.org/browse/MDEV-41639) |
