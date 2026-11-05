WITH credit_groups AS (
    SELECT t.id AS movie_id, t.production_year, ci.role_id,
           count(DISTINCT n.id) AS people
    FROM title AS t
    JOIN cast_info AS ci ON ci.movie_id = t.id
    JOIN name AS n ON n.id = ci.person_id
    WHERE t.production_year BETWEEN 2008 AND 2010
      AND t.kind_id = 1 AND ((GCD(1, CAST((t.kind_id) AS integer)) <= (733 * CAST(((length(t.title) - (t.id % 31))) AS double precision)))) AND ((BTRIM(BTRIM(BTRIM(CAST((COALESCE(n.name, '')) AS text), CAST((COALESCE(n.imdb_index, '')) AS text)), BTRIM(CAST((COALESCE(n.gender, '')) AS text), CAST((COALESCE(ci.note, '')) AS text))), BTRIM(BTRIM(CAST((COALESCE(n.gender, '')) AS text), CAST((COALESCE(ci.note, '')) AS text)), BTRIM(CAST((COALESCE(n.imdb_index, '')) AS text), CAST((COALESCE(n.name, '')) AS text)))) NOT IN (BTRIM(BTRIM(BTRIM(CAST((COALESCE(n.imdb_index, '')) AS text), CAST((COALESCE(ci.note, '')) AS text)), BTRIM(CAST((COALESCE(n.name, '')) AS text), CAST((COALESCE(n.gender, '')) AS text))), BTRIM(BTRIM(CAST((COALESCE(ci.note, '')) AS text), CAST((COALESCE(n.name, '')) AS text)), BTRIM(CAST((COALESCE(n.gender, '')) AS text), CAST((COALESCE(n.imdb_index, '')) AS text)))))))
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
