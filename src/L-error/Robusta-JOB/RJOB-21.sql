-- Kalita-JOB feedback-guided: GJOB_T34; variation 1; focused_cohort_v1
SELECT kt.kind, t.production_year, n.gender,
       count(DISTINCT t.id) AS movies, count(DISTINCT n.id) AS people,
       count(*) AS qualifying_credits
FROM title t JOIN kind_type kt ON kt.id = t.kind_id
JOIN cast_info ci ON ci.movie_id = t.id JOIN name n ON n.id = ci.person_id
WHERE t.production_year = 2009 AND ci.role_id IN (1, 2) AND t.kind_id IN (1, 3)
  AND ((CHAR_LENGTH(CAST((t.title) AS text)) != CAST((length(n.name)) AS numeric))) AND ((CAST((length(t.title)) AS numeric) != OCTET_LENGTH(LOWER(CAST((n.name) AS text)))))
  AND EXISTS (SELECT 1 FROM movie_keyword mk JOIN keyword k ON k.id = mk.keyword_id
              WHERE mk.movie_id = t.id AND k.keyword IN ('sequel', 'horror'))
  AND EXISTS (SELECT 1 FROM person_info pi JOIN info_type it ON it.id = pi.info_type_id
              WHERE pi.person_id = n.id AND pi.info IS NOT NULL AND pi.info_type_id = 21)
  AND NOT EXISTS (SELECT 1 FROM aka_name an
                  WHERE an.person_id = n.id AND an.name = n.name)
GROUP BY kt.kind, t.production_year, n.gender
