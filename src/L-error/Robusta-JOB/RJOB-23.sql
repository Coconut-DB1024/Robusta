-- Kalita-JOB feedback-guided: GJOB_T36; variation 2; full_measurement_feedback_v2
WITH experienced AS (
    SELECT rt.role, cn.country_code, t.production_year,
           count(DISTINCT n.id) AS people, count(DISTINCT t.id) AS movies
    FROM title t JOIN cast_info ci ON ci.movie_id = t.id
    JOIN name n ON n.id = ci.person_id JOIN role_type rt ON rt.id = ci.role_id
    JOIN movie_companies mc ON mc.movie_id = t.id
    JOIN company_name cn ON cn.id = mc.company_id
    JOIN company_type ct ON ct.id = mc.company_type_id
    WHERE t.production_year BETWEEN 2008 AND 2009
      AND ci.role_id IN (1, 2) AND cn.country_code IN ('[us]', '[gb]')
      AND ct.kind = 'production companies' AND t.kind_id = 1
      AND ((CAST((length(n.name)) AS numeric) < LENGTH(CAST((t.title) AS text)))) AND (((CAST((lower(t.title)) AS text) = CAST((lower(n.name)) AS text)) OR (CAST((lower(t.title)) AS text) >= CAST((lower(n.name)) AS text))))
      AND EXISTS (SELECT 1 FROM cast_info earlier_ci
                  JOIN title earlier ON earlier.id = earlier_ci.movie_id
                  WHERE earlier_ci.person_id = n.id AND earlier_ci.role_id = ci.role_id
                    AND earlier.kind_id = 1
                    AND earlier.production_year BETWEEN 1990 AND t.production_year - 1)
    GROUP BY rt.role, cn.country_code, t.production_year
)
SELECT *, dense_rank() OVER (PARTITION BY role, production_year ORDER BY people DESC) AS country_rank,
       avg(people) OVER (PARTITION BY role, country_code) AS mean_annual_people
FROM experienced
