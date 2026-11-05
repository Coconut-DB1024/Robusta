-- Kalita-JOB feedback-guided: GJOB_T02; variation 1; initial
SELECT lt.link, kt.kind, it.info, t.production_year,
       count(DISTINCT ml.id) AS movie_links,
       avg(linked.production_year - t.production_year) AS mean_year_gap
FROM movie_link ml
JOIN title t ON t.id = ml.movie_id
JOIN title linked ON linked.id = ml.linked_movie_id
JOIN link_type lt ON lt.id = ml.link_type_id
JOIN kind_type kt ON kt.id = t.kind_id
JOIN movie_info_idx mii ON mii.movie_id = t.id
JOIN info_type it ON it.id = mii.info_type_id
WHERE t.production_year BETWEEN 1998 AND 2010
  AND linked.production_year >= t.production_year
  AND ((CAST(((linked.production_year - 1900)) AS numeric) != LENGTH(CAST((linked.title) AS text)))) AND ((CAST((length(t.title)) AS numeric) != OCTET_LENGTH(CAST((mii.info) AS text))))
  AND EXISTS (SELECT 1 FROM cast_info ci
              JOIN name n ON n.id = ci.person_id
              WHERE ci.movie_id = linked.id AND n.gender IS NOT NULL)
GROUP BY lt.link, kt.kind, it.info, t.production_year
