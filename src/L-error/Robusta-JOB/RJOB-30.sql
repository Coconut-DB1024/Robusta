SELECT t.kind_id, rt.role, count(DISTINCT ci.person_id) AS aliased_people,
       count(DISTINCT t.id) AS movies, count(*) AS credits
FROM title AS t
JOIN cast_info AS ci ON ci.movie_id = t.id
JOIN role_type AS rt ON rt.id = ci.role_id
WHERE t.production_year BETWEEN 2005 AND 2010
  AND t.kind_id IN (1, 3) AND (((CAST((t.kind_id) AS bigint) * CAST((length(t.title)) AS smallint)) NOT IN (CEILING(7.17))))
  AND EXISTS (
      SELECT 1
      FROM aka_name AS an
      JOIN name AS n ON n.id = an.person_id
      WHERE an.person_id = ci.person_id AND ((BTRIM(BTRIM(BTRIM(CAST((COALESCE(an.name, '')) AS text), CAST((COALESCE(an.imdb_index, '')) AS text)), BTRIM(CAST((COALESCE(n.name, '')) AS text), CAST((COALESCE(n.imdb_index, '')) AS text))), BTRIM(BTRIM(CAST((COALESCE(n.name, '')) AS text), CAST((COALESCE(n.imdb_index, '')) AS text)), BTRIM(CAST((COALESCE(an.imdb_index, '')) AS text), CAST((COALESCE(an.name, '')) AS text)))) IS DISTINCT FROM BTRIM(BTRIM(BTRIM(CAST((COALESCE(an.imdb_index, '')) AS text), CAST((COALESCE(n.imdb_index, '')) AS text)), BTRIM(CAST((COALESCE(an.name, '')) AS text), CAST((COALESCE(n.name, '')) AS text))), BTRIM(BTRIM(CAST((COALESCE(n.imdb_index, '')) AS text), CAST((COALESCE(an.name, '')) AS text)), BTRIM(CAST((COALESCE(n.name, '')) AS text), CAST((COALESCE(an.imdb_index, '')) AS text))))))
        AND (an.name <> n.name OR an.imdb_index IS NOT NULL)
  )
GROUP BY t.kind_id, rt.role
HAVING count(DISTINCT ci.person_id) >= 1
