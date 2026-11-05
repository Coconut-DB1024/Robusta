-- Kalita-JOB feedback-guided: GJOB_T21; variation 1; focused_cohort_v1
SELECT rt.role, cn.country_code, k.phonetic_code,
       count(DISTINCT t.id) AS movies,
       count(DISTINCT n.id) AS credited_people,
       count(DISTINCT cn.id) AS companies
FROM title t
JOIN cast_info ci ON ci.movie_id = t.id
JOIN name n ON n.id = ci.person_id
JOIN role_type rt ON rt.id = ci.role_id
JOIN movie_companies mc ON mc.movie_id = t.id
JOIN company_name cn ON cn.id = mc.company_id
JOIN company_type ct ON ct.id = mc.company_type_id
JOIN movie_keyword mk ON mk.movie_id = t.id
JOIN keyword k ON k.id = mk.keyword_id
WHERE t.production_year = 2009 AND t.kind_id = 1
  AND cn.country_code = '[us]'
  AND ct.kind = 'production companies'
  AND ci.role_id IN (1, 2)
  AND k.keyword IN ('sequel', 'horror')
  AND ((CHAR_LENGTH(CAST((t.title) AS text)) != CAST((length(cn.name)) AS numeric))) AND ((CAST((length(t.title)) AS numeric) != OCTET_LENGTH(LOWER(CAST((n.name) AS text)))))
GROUP BY rt.role, cn.country_code, k.phonetic_code
HAVING count(DISTINCT t.id) > 0
