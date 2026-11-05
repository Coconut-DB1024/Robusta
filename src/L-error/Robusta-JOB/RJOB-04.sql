-- Kalita-JOB feedback-guided: GJOB_T10; variation 1; initial
SELECT parent.title AS series_title, kt.kind, status.kind AS cast_status, rt.role,
       count(DISTINCT ep.id) AS episodes, count(DISTINCT n.id) AS people
FROM title ep
JOIN title parent ON parent.id = ep.episode_of_id
JOIN kind_type kt ON kt.id = ep.kind_id
JOIN cast_info ci ON ci.movie_id = ep.id
JOIN name n ON n.id = ci.person_id
JOIN role_type rt ON rt.id = ci.role_id
JOIN complete_cast cc ON cc.movie_id = ep.id
JOIN comp_cast_type status ON status.id = cc.status_id
WHERE ep.production_year BETWEEN 2003 AND 2010
  AND ((CAST(((parent.production_year - 1900)) AS numeric) != LENGTH(CAST((parent.title) AS text)))) AND ((CAST((ci.nr_order) AS numeric) != OCTET_LENGTH(CAST((n.name) AS text))))
  AND EXISTS (SELECT 1 FROM aka_title alternate WHERE alternate.movie_id = parent.id)
GROUP BY parent.id, parent.title, kt.kind, status.kind, rt.role
