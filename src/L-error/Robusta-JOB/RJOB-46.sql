SELECT t.production_year,k.keyword,n.gender,rt.role,
 count(DISTINCT t.id) AS movies,count(DISTINCT n.id) AS people,
 count(DISTINCT ch.id) AS characters
 FROM title t JOIN cast_info ci ON ci.movie_id=t.id JOIN name n ON n.id=ci.person_id
 JOIN char_name ch ON ch.id=ci.person_role_id JOIN role_type rt ON rt.id=ci.role_id
 JOIN movie_keyword mk ON mk.movie_id=t.id JOIN keyword k ON k.id=mk.keyword_id
 WHERE t.production_year BETWEEN 1988 AND 2012 AND t.kind_id=1
 AND ci.role_id IN (1,2) AND ((CHAR_LENGTH(CAST((t.title) AS character varying)) / (CAST((length(n.name)) AS integer) / 1)) != (STRPOS(CAST((t.title) AS character), CAST((n.name) AS character)) % (612 + 678)))
 AND k.keyword IN ('sequel','horror','violence')
 AND EXISTS (SELECT 1 FROM aka_name an WHERE an.person_id=n.id AND an.name<>n.name)
 AND EXISTS (SELECT 1 FROM movie_companies mc JOIN company_name cn ON cn.id=mc.company_id WHERE mc.movie_id=t.id AND cn.country_code IN ('[us]','[gb]'))
 GROUP BY t.production_year,k.keyword,n.gender,rt.role;
