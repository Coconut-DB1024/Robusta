-- Robusta Guided V5 V5_THEMES_01: Gender/country workforce of rated sequel/horror movies, with actual rating membership and within-year workforce ranks.
WITH workforce AS (
 SELECT k.keyword,rt.role,cn.country_code,t.production_year,n.gender,
 count(DISTINCT t.id) AS movies,count(DISTINCT n.id) AS credited_people,
 count(DISTINCT cn.id) AS production_companies
 FROM title t JOIN cast_info ci ON ci.movie_id=t.id
 JOIN name n ON n.id=ci.person_id JOIN role_type rt ON rt.id=ci.role_id
 JOIN movie_companies mc ON mc.movie_id=t.id
 JOIN company_name cn ON cn.id=mc.company_id JOIN company_type ct ON ct.id=mc.company_type_id
 JOIN movie_keyword mk ON mk.movie_id=t.id JOIN keyword k ON k.id=mk.keyword_id
 WHERE t.production_year BETWEEN 2002 AND 2009 AND t.kind_id=1
 AND ci.role_id IN (1,2) AND cn.country_code IN ('[us]','[gb]')
 AND ct.kind='production companies' AND k.keyword IN ('sequel','horror') AND (CAST((length(n.name)) AS integer) < CAST((length(t.title)) AS integer))
 AND EXISTS (SELECT 1 FROM movie_info_idx rating WHERE rating.movie_id=t.id AND rating.info_type_id=101 AND rating.info>'6.0')
 GROUP BY k.keyword,rt.role,cn.country_code,t.production_year,n.gender
)
SELECT *,credited_people::numeric/NULLIF(movies,0) AS people_per_movie,
 dense_rank() OVER (PARTITION BY keyword,role,production_year ORDER BY credited_people DESC) AS workforce_rank
FROM workforce;
