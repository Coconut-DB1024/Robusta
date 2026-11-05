WITH credit_groups AS (
    SELECT t.id AS movie_id, t.production_year, ci.role_id,
           count(DISTINCT n.id) AS people
    FROM title AS t
    JOIN cast_info AS ci ON ci.movie_id = t.id
    JOIN name AS n ON n.id = ci.person_id
    WHERE t.production_year BETWEEN 2008 AND 2010
      AND t.kind_id = 1 AND ((ABS(CAST(((t.id % 997)) AS numeric)) <= (177 * CAST((t.kind_id) AS smallint)))) AND ((BTRIM(CAST((COALESCE(n.name, '')) AS text), CAST((COALESCE(n.imdb_index, '')) AS text)) > BTRIM(CAST((COALESCE(n.gender, '')) AS text), CAST((COALESCE(ci.note, '')) AS text))))
    GROUP BY t.id, t.production_year, ci.role_id
), ranked AS (
    SELECT cg.*, dense_rank() OVER (
        PARTITION BY cg.role_id ORDER BY cg.people DESC
    ) AS size_rank
    FROM credit_groups AS cg
)
SELECT rt.role, ranked.production_year,
       count(*) AS movie_role_groups, sum(ranked.people) AS total_people,
       max(ranked.size_rank) AS distinct_credit_sizes
FROM ranked
JOIN role_type AS rt ON rt.id = ranked.role_id
GROUP BY rt.role, ranked.production_year
