SELECT t.production_year, rt.role, n.gender,
       count(*) AS credits, count(DISTINCT n.id) AS people,
       count(DISTINCT t.id) AS movies
FROM title AS t
JOIN cast_info AS ci ON ci.movie_id = t.id
JOIN name AS n ON n.id = ci.person_id
JOIN role_type AS rt ON rt.id = ci.role_id
WHERE t.production_year BETWEEN 2005 AND 2010
  AND t.kind_id = 1 AND (((CAST(((t.id % 997)) AS smallint) - CAST((length(t.title)) AS smallint)) < GCD(CAST(((t.production_year - 1900)) AS integer), CAST((t.kind_id) AS smallint)))) AND ((BTRIM(BTRIM(CAST((COALESCE(n.name, '')) AS text), CAST((COALESCE(n.imdb_index, '')) AS text)), BTRIM(CAST((COALESCE(n.gender, '')) AS text), CAST((COALESCE(ci.note, '')) AS text))) NOT IN (BTRIM(BTRIM(CAST((COALESCE(n.gender, '')) AS text), CAST((COALESCE(ci.note, '')) AS text)), BTRIM(CAST((COALESCE(n.imdb_index, '')) AS text), CAST((COALESCE(n.name, '')) AS text))))))
  AND (ci.nr_order IS NOT NULL OR ci.note IS NOT NULL)
GROUP BY t.production_year, rt.role, n.gender
HAVING count(*) >= count(DISTINCT t.id)
