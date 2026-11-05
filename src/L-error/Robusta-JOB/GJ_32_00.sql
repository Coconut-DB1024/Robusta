-- Kalita-JOB feedback-guided: GJOB_T32; variation 0; focused_cohort_v1
WITH eligible AS (
    SELECT t.id, t.kind_id, t.production_year
    FROM title t
    WHERE t.production_year = 2009 AND t.kind_id IN (1, 3)
      AND ((CHAR_LENGTH('typical_str') != CAST((length(t.title)) AS numeric))) AND (((CAST((lower(t.title)) AS text) < 'typical_str') AND (CAST((lower(t.title)) AS text) > 'test_data')))
      AND EXISTS (SELECT 1 FROM movie_keyword mk JOIN keyword k ON k.id = mk.keyword_id
                  WHERE mk.movie_id = t.id AND k.keyword IN ('sequel', 'horror'))
), facts AS (
    SELECT e.id AS movie_id, count(DISTINCT it.id) AS types
    FROM eligible e JOIN movie_info mi ON mi.movie_id = e.id
    JOIN info_type it ON it.id = mi.info_type_id GROUP BY e.id
), companies AS (
    SELECT e.id AS movie_id, count(DISTINCT cn.id) AS companies,
           count(DISTINCT cn.country_code) AS countries
    FROM eligible e JOIN movie_companies mc ON mc.movie_id = e.id
    JOIN company_name cn ON cn.id = mc.company_id
    JOIN company_type ct ON ct.id = mc.company_type_id
    WHERE cn.country_code = '[us]' AND ct.kind = 'production companies'
    GROUP BY e.id
)
SELECT kt.kind, e.production_year, count(*) AS movies,
       avg(f.types::numeric / c.companies) AS info_company_ratio,
       sum(c.countries) AS country_coverage
FROM eligible e JOIN facts f ON f.movie_id = e.id
JOIN companies c ON c.movie_id = e.id JOIN kind_type kt ON kt.id = e.kind_id
GROUP BY kt.kind, e.production_year
