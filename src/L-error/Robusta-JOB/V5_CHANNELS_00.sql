-- Robusta Guided V5 V5_CHANNELS_00: Yearly actor and production-company coverage against all eligible titles, including titles without a qualifying credit; denominator CTE is required for coverage.
WITH eligible AS (
 SELECT t.id,t.production_year,kt.kind FROM title t JOIN kind_type kt ON kt.id=t.kind_id
 WHERE t.production_year BETWEEN 1998 AND 2010 AND t.kind_id IN (1,3)
 AND (CAST((length(kt.kind)) AS integer) < CAST((length(t.title)) AS integer)) AND lower(t.title)>'test_data' AND lower(t.title)<'typical_str'
), totals AS (
 SELECT kind,production_year,count(*) AS cohort_titles FROM eligible GROUP BY kind,production_year
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
 JOIN company_name cn ON cn.id=mc.company_id JOIN company_type ct ON ct.id=mc.company_type_id
 WHERE cn.country_code='[us]' AND ct.kind='production companies'
 GROUP BY e.kind,e.production_year
)
SELECT r.*,c.cohort_titles,r.movies::numeric/NULLIF(c.cohort_titles,0) AS coverage,
 r.entities::numeric/NULLIF(r.movies,0) AS entities_per_covered_title,
 lag(r.entities) OVER (PARTITION BY r.kind,r.channel ORDER BY r.production_year) AS previous_cohort_entities
FROM channels r JOIN totals c USING(kind,production_year);
