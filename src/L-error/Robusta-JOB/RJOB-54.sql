-- Guided v2 V2_CHANNELS_01: Two distinct people/company channels, with year-over-year cohort entity counts; CTE reuse is functional.
WITH eligible AS (
 SELECT t.id,t.production_year,kt.kind FROM title t JOIN kind_type kt ON kt.id=t.kind_id
 WHERE t.production_year BETWEEN 2005 AND 2010 AND t.kind_id IN (1,3)
 AND (CAST((length(kt.kind)) AS integer) < CAST((length(t.title)) AS integer)) AND lower(t.title)>'test_data' AND lower(t.title)<'typical_str'
), channels AS (
 SELECT e.kind,e.production_year,'people'::text AS channel,
 count(DISTINCT e.id) AS movies,count(DISTINCT n.id) AS entities
 FROM eligible e JOIN cast_info ci ON ci.movie_id=e.id
 JOIN name n ON n.id=ci.person_id JOIN role_type rt ON rt.id=ci.role_id
 WHERE ci.role_id IN (1,2) GROUP BY e.kind,e.production_year
 UNION ALL
 SELECT e.kind,e.production_year,'companies'::text AS channel,
 count(DISTINCT e.id) AS movies,count(DISTINCT cn.id) AS entities
 FROM eligible e JOIN movie_companies mc ON mc.movie_id=e.id
 JOIN company_name cn ON cn.id=mc.company_id
 JOIN company_type ct ON ct.id=mc.company_type_id
 WHERE cn.country_code='[us]' AND ct.kind='production companies'
 GROUP BY e.kind,e.production_year)
SELECT *,entities::numeric/NULLIF(movies,0) AS entities_per_movie,
 lag(entities) OVER (PARTITION BY kind,channel ORDER BY production_year) AS previous_cohort_entities
FROM channels;
