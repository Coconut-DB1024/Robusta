-- Guided v3 V3_THEMES_02: Theme-specific production workforce: year-level and all-years rollups share the same genuine joins; each table supplies a dimension or eligibility condition.
SELECT k.keyword,rt.role,cn.country_code,t.production_year,
 GROUPING(t.production_year) AS is_all_years,
 count(DISTINCT t.id) AS movies,count(DISTINCT n.id) AS credited_people,
 count(DISTINCT cn.id) AS production_companies
FROM title t JOIN cast_info ci ON ci.movie_id=t.id
JOIN name n ON n.id=ci.person_id JOIN role_type rt ON rt.id=ci.role_id
JOIN movie_companies mc ON mc.movie_id=t.id
JOIN company_name cn ON cn.id=mc.company_id
JOIN company_type ct ON ct.id=mc.company_type_id
JOIN movie_keyword mk ON mk.movie_id=t.id JOIN keyword k ON k.id=mk.keyword_id
WHERE t.production_year BETWEEN 2006 AND 2009 AND t.kind_id=1
AND ci.role_id IN (1,2) AND cn.country_code IN ('[us]','[gb]')
AND ct.kind='production companies' AND k.keyword IN ('sequel','horror')
AND (CAST((length(n.name)) AS integer) < CAST((length(t.title)) AS integer)) AND (LOWER(CAST((cn.name) AS character varying)) <= ('typical_str' || CAST((t.production_year) AS numeric)))
GROUP BY GROUPING SETS
 ((k.keyword,rt.role,cn.country_code,t.production_year),(k.keyword,rt.role,cn.country_code));
