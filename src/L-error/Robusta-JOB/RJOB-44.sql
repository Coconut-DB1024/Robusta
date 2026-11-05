SELECT rt.role, t.kind_id,
       count(DISTINCT ch.id) AS named_characters,
       count(DISTINCT ci.person_id) AS documented_people,
       count(DISTINCT t.id) AS movies, count(*) AS credits
FROM title AS t
JOIN cast_info AS ci ON ci.movie_id = t.id
JOIN char_name AS ch ON ch.id = ci.person_role_id
JOIN role_type AS rt ON rt.id = ci.role_id
WHERE t.production_year BETWEEN 2000 AND 2010
  AND t.kind_id IN (1, 3) AND (((CAST(((t.id % 997)) AS numeric) - CAST((length(t.title)) AS numeric)) NOT IN (GCD(CAST(((t.production_year - 1900)) AS numeric), CAST((t.kind_id) AS numeric))))) AND ((BTRIM(BTRIM(CAST((COALESCE(ch.name, '')) AS text), CAST((COALESCE(ch.imdb_index, '')) AS text)), BTRIM(CAST((COALESCE(ch.name_pcode_nf, '')) AS text), CAST((COALESCE(ch.surname_pcode, '')) AS text))) >= BTRIM(BTRIM(CAST((COALESCE(ch.name_pcode_nf, '')) AS text), CAST((COALESCE(ch.surname_pcode, '')) AS text)), BTRIM(CAST((COALESCE(ch.imdb_index, '')) AS text), CAST((COALESCE(ch.name, '')) AS text)))))
  AND EXISTS (
      SELECT 1 FROM person_info AS pi
      JOIN info_type AS it ON it.id = pi.info_type_id
      WHERE pi.person_id = ci.person_id
        AND (pi.note IS NOT NULL OR length(pi.info) >= length(it.info))
  )
GROUP BY rt.role, t.kind_id
