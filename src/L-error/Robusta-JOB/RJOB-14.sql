-- Kalita-JOB feedback-guided: GJOB_T28; variation 0; focused_cohort_v1
SELECT cn.country_code, it.info, t.production_year,
       count(DISTINCT t.id) AS movies, count(DISTINCT mi.id) AS unindexed_facts,
       count(DISTINCT k.id) AS keywords
FROM title t
JOIN movie_info mi ON mi.movie_id = t.id
JOIN info_type it ON it.id = mi.info_type_id
JOIN movie_companies mc ON mc.movie_id = t.id
JOIN company_name cn ON cn.id = mc.company_id
JOIN company_type ct ON ct.id = mc.company_type_id
JOIN movie_keyword mk ON mk.movie_id = t.id
JOIN keyword k ON k.id = mk.keyword_id
WHERE t.production_year = 2009
  AND cn.country_code = '[us]' AND ct.kind = 'production companies'
  AND k.keyword IN ('sequel', 'horror') AND mi.info_type_id IN (3, 8) AND t.kind_id = 1
  AND ((CAST((length(t.title)) AS numeric) < LENGTH(CAST((cn.name) AS text)))) AND ((CHAR_LENGTH(UPPER(CAST((t.title) AS text))) != CAST((length(cn.name)) AS numeric)))
  AND NOT EXISTS (SELECT 1 FROM movie_info_idx indexed
                  WHERE indexed.movie_id = mi.movie_id
                    AND indexed.info_type_id = mi.info_type_id)
GROUP BY cn.country_code, it.info, t.production_year
