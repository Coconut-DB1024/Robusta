-- Kalita-JOB feedback-guided: GJOB_T15; variation 0; initial
SELECT lt.link, cn.country_code, it.info,
       count(DISTINCT ml.id) AS movie_links,
       count(DISTINCT t.id) AS movies,
       avg(linked.production_year - t.production_year) AS mean_year_gap
FROM title t JOIN movie_link ml ON ml.movie_id = t.id
JOIN title linked ON linked.id = ml.linked_movie_id
JOIN link_type lt ON lt.id = ml.link_type_id
JOIN movie_companies mc ON mc.movie_id = linked.id
JOIN company_name cn ON cn.id = mc.company_id
JOIN movie_info mi ON mi.movie_id = t.id JOIN info_type it ON it.id = mi.info_type_id
WHERE t.production_year BETWEEN 1998 AND 2010
  AND linked.production_year >= t.production_year
  AND ((CAST(((t.production_year - 1900)) AS numeric) != LENGTH(UPPER(CAST((t.title) AS text))))) AND ((CAST(((linked.production_year - 1900)) AS numeric) != OCTET_LENGTH(LOWER(CAST((linked.title) AS text)))))
  AND NOT EXISTS (SELECT 1 FROM movie_info_idx indexed
                  WHERE indexed.movie_id = mi.movie_id AND indexed.info_type_id = mi.info_type_id)
GROUP BY lt.link, cn.country_code, it.info
