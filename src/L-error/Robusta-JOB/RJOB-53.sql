-- Guided v2 V2_CAREER_01: Country/year production analysis of experienced actors; distinct companies and people/movie ratio.
WITH experienced AS (
 SELECT rt.role, cn.country_code, t.production_year,
        count(DISTINCT n.id) AS people, count(DISTINCT t.id) AS movies,
        count(DISTINCT cn.id) AS production_companies
 FROM title t JOIN cast_info ci ON ci.movie_id=t.id
 JOIN name n ON n.id=ci.person_id JOIN role_type rt ON rt.id=ci.role_id
 JOIN movie_companies mc ON mc.movie_id=t.id
 JOIN company_name cn ON cn.id=mc.company_id
 JOIN company_type ct ON ct.id=mc.company_type_id
 WHERE t.production_year BETWEEN 2006 AND 2009
 AND ci.role_id IN (1,2) AND cn.country_code IN ('[us]','[gb]')
 AND ct.kind='production companies' AND t.kind_id=1
 AND (CAST((length(n.name)) AS integer) < CAST((length(t.title)) AS integer)) AND lower(t.title)>=lower(n.name)
 AND EXISTS (SELECT 1 FROM cast_info earlier_ci
 JOIN title earlier ON earlier.id=earlier_ci.movie_id
 WHERE earlier_ci.person_id=n.id AND earlier_ci.role_id=ci.role_id
 AND earlier.kind_id=1 AND earlier.production_year BETWEEN 1990 AND t.production_year-1)
 GROUP BY rt.role,cn.country_code,t.production_year)
SELECT *, dense_rank() OVER (PARTITION BY role,production_year ORDER BY people DESC) AS country_rank,
 people::numeric/NULLIF(movies,0) AS distinct_people_per_movie
FROM experienced;
