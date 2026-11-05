-- Kalita-JOB feedback-guided: GJOB_T27; variation 1; focused_cohort_v1
SELECT rt.role, ch.name AS character_name, n.gender,
       count(DISTINCT t.id) AS movies,
       count(DISTINCT n.id) AS people, avg(ci.nr_order) AS mean_credit_order
FROM title t
JOIN cast_info ci ON ci.movie_id = t.id
JOIN name n ON n.id = ci.person_id
JOIN char_name ch ON ch.id = ci.person_role_id
JOIN role_type rt ON rt.id = ci.role_id
WHERE t.production_year BETWEEN 2008 AND 2009
  AND ci.role_id IN (1, 2) AND t.kind_id = 1
  AND ((CHAR_LENGTH(CAST((t.title) AS text)) != CAST((length(n.name)) AS numeric))) AND ((CAST((length(t.title)) AS numeric) != OCTET_LENGTH(LOWER(CAST((ch.name) AS text)))))
  AND EXISTS (SELECT 1 FROM aka_name an
              WHERE an.person_id = n.id AND an.name <> n.name)
  AND EXISTS (SELECT 1 FROM person_info pi
              JOIN info_type it ON it.id = pi.info_type_id
              WHERE pi.person_id = n.id AND pi.info IS NOT NULL AND pi.info_type_id = 21)
GROUP BY rt.role, ch.id, ch.name, n.gender
