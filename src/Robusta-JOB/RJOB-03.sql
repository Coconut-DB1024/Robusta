-- Kalita-JOB feedback-guided: GJOB_T09; variation 2; full_measurement_feedback_v2
SELECT k1.phonetic_code AS first_family, k2.phonetic_code AS second_family,
       t.production_year, count(DISTINCT t.id) AS movies,
       count(*) AS observed_keyword_pairs
FROM title t
JOIN movie_keyword mk1 ON mk1.movie_id = t.id
JOIN keyword k1 ON k1.id = mk1.keyword_id
JOIN movie_keyword mk2 ON mk2.movie_id = t.id AND mk2.keyword_id > mk1.keyword_id
JOIN keyword k2 ON k2.id = mk2.keyword_id
WHERE t.production_year BETWEEN 2008 AND 2010 AND t.kind_id = 1
  AND ((CAST((length(k2.keyword)) AS numeric) < LENGTH(CAST((t.title) AS text)))) AND (((CAST((lower(t.title)) AS text) = CAST((lower(k2.keyword)) AS text)) OR (CAST((lower(t.title)) AS text) >= CAST((lower(k2.keyword)) AS text))))
  AND EXISTS (SELECT 1 FROM movie_companies mc
              JOIN company_name cn ON cn.id = mc.company_id
              WHERE mc.movie_id = t.id AND cn.country_code IS NOT NULL)
GROUP BY k1.phonetic_code, k2.phonetic_code, t.production_year
