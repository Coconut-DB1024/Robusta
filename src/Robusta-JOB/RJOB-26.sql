SELECT k.phonetic_code, t.kind_id,
       count(*) AS keyword_mentions,
       count(DISTINCT k.id) AS keywords,
       count(DISTINCT t.id) AS movies
FROM title AS t
JOIN movie_keyword AS mk ON mk.movie_id = t.id
JOIN keyword AS k ON k.id = mk.keyword_id
WHERE t.production_year BETWEEN 2000 AND 2010
  AND t.kind_id IN (1, 3, 4) AND (((CAST(((t.id % 997)) AS smallint) - CAST((length(t.title)) AS smallint)) <= GCD(CAST(((t.production_year - 1900)) AS integer), CAST((t.kind_id) AS smallint)))) AND ((BTRIM(BTRIM(CAST((COALESCE(k.keyword, '')) AS text), CAST((COALESCE(k.phonetic_code, '')) AS text)), BTRIM(CAST((COALESCE(t.title, '')) AS text), CAST((COALESCE(t.imdb_index, '')) AS text))) != BTRIM(BTRIM(CAST((COALESCE(t.title, '')) AS text), CAST((COALESCE(t.imdb_index, '')) AS text)), BTRIM(CAST((COALESCE(k.phonetic_code, '')) AS text), CAST((COALESCE(k.keyword, '')) AS text)))))
GROUP BY k.phonetic_code, t.kind_id
HAVING count(DISTINCT t.id) >= 1
