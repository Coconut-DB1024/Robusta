SELECT rt.role, n.gender, count(DISTINCT ci.person_id) AS people,
       count(DISTINCT t.id) AS current_movies, count(*) AS credits
FROM title AS t
JOIN cast_info AS ci ON ci.movie_id = t.id
JOIN name AS n ON n.id = ci.person_id
JOIN role_type AS rt ON rt.id = ci.role_id
WHERE t.production_year BETWEEN 2008 AND 2010
  AND t.kind_id = 1 AND ((GCD(1, CAST((t.kind_id) AS integer)) < (390 * CAST(((length(t.title) - (t.id % 31))) AS double precision)))) AND ((BTRIM(BTRIM(CAST((COALESCE(n.name, '')) AS text), CAST((COALESCE(n.imdb_index, '')) AS text)), BTRIM(CAST((COALESCE(n.gender, '')) AS text), CAST((COALESCE(ci.note, '')) AS text))) != BTRIM(BTRIM(CAST((COALESCE(n.gender, '')) AS text), CAST((COALESCE(ci.note, '')) AS text)), BTRIM(CAST((COALESCE(n.imdb_index, '')) AS text), CAST((COALESCE(n.name, '')) AS text)))))
  AND EXISTS (
      SELECT 1 FROM cast_info AS earlier_credit
      JOIN title AS earlier ON earlier.id = earlier_credit.movie_id
      WHERE earlier_credit.person_id = ci.person_id
        AND earlier_credit.role_id = ci.role_id
        AND earlier.kind_id = 1
        AND earlier.production_year BETWEEN 1990 AND 2007
        AND earlier.production_year < t.production_year
  )
GROUP BY rt.role, n.gender
