SELECT it.info, t.kind_id, count(*) AS unindexed_information_rows,
       count(DISTINCT t.id) AS movies
FROM title AS t
JOIN movie_info AS mi ON mi.movie_id = t.id
JOIN info_type AS it ON it.id = mi.info_type_id
WHERE t.production_year BETWEEN 2005 AND 2010
  AND t.kind_id = 1 AND (((CAST(((t.id % 997)) AS numeric) - CAST((length(t.title)) AS smallint)) < OCTET_LENGTH('test_data'))) AND ((BTRIM(BTRIM(BTRIM(CAST((COALESCE(mi.info, '')) AS text), CAST((COALESCE(mi.note, '')) AS text)), BTRIM(CAST((COALESCE(it.info, '')) AS text), CAST((COALESCE(t.title, '')) AS text))), BTRIM(BTRIM(CAST((COALESCE(it.info, '')) AS text), CAST((COALESCE(t.title, '')) AS text)), BTRIM(CAST((COALESCE(mi.note, '')) AS text), CAST((COALESCE(mi.info, '')) AS text)))) <= BTRIM(BTRIM(BTRIM(CAST((COALESCE(mi.note, '')) AS text), CAST((COALESCE(t.title, '')) AS text)), BTRIM(CAST((COALESCE(mi.info, '')) AS text), CAST((COALESCE(it.info, '')) AS text))), BTRIM(BTRIM(CAST((COALESCE(t.title, '')) AS text), CAST((COALESCE(mi.info, '')) AS text)), BTRIM(CAST((COALESCE(it.info, '')) AS text), CAST((COALESCE(mi.note, '')) AS text))))))
  AND NOT EXISTS (
      SELECT 1 FROM movie_info_idx AS indexed
      WHERE indexed.movie_id = t.id AND indexed.info_type_id = mi.info_type_id
  )
GROUP BY it.info, t.kind_id
