-- Kalita-JOB feedback-guided: GJOB_T33; variation 0; focused_cohort_v1
WITH eligible AS (
    SELECT t.id, t.production_year, kt.kind FROM title t
    JOIN kind_type kt ON kt.id = t.kind_id
    WHERE t.production_year = 2009 AND t.kind_id IN (1, 3)
      AND ((CAST((length(kt.kind)) AS numeric) < LENGTH(CAST((t.title) AS text)))) AND (((CAST((lower(t.title)) AS text) < 'typical_str') AND (CAST((lower(t.title)) AS text) > 'test_data')))
), channels AS (
    SELECT e.kind, e.production_year, 'people'::text AS channel,
           count(DISTINCT e.id) AS movies, count(DISTINCT n.id) AS entities
    FROM eligible e JOIN cast_info ci ON ci.movie_id = e.id
    JOIN name n ON n.id = ci.person_id JOIN role_type rt ON rt.id = ci.role_id
    WHERE ci.role_id IN (1, 2)
    GROUP BY e.kind, e.production_year
    UNION ALL
    SELECT e.kind, e.production_year, 'companies'::text AS channel,
           count(DISTINCT e.id) AS movies, count(DISTINCT cn.id) AS entities
    FROM eligible e JOIN movie_companies mc ON mc.movie_id = e.id
    JOIN company_name cn ON cn.id = mc.company_id
    JOIN company_type ct ON ct.id = mc.company_type_id
    WHERE cn.country_code = '[us]' AND ct.kind = 'production companies'
    GROUP BY e.kind, e.production_year
)
SELECT kind, production_year,
       sum(movies) FILTER (WHERE channel = 'people') AS cast_movies,
       sum(movies) FILTER (WHERE channel = 'companies') AS company_movies,
       sum(entities) FILTER (WHERE channel = 'people') AS people,
       sum(entities) FILTER (WHERE channel = 'companies') AS companies
FROM channels GROUP BY kind, production_year
