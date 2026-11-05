WITH coverage AS (
    SELECT t.production_year, kt.kind AS movie_kind, cn.country_code,
           ct.kind AS company_role, count(DISTINCT t.id) AS movies,
           count(DISTINCT k.id) AS keyword_count, count(*) AS company_keyword_mentions
    FROM title AS t
    JOIN kind_type AS kt ON kt.id = t.kind_id
    JOIN movie_companies AS mc ON mc.movie_id = t.id
    JOIN company_name AS cn ON cn.id = mc.company_id
    JOIN company_type AS ct ON ct.id = mc.company_type_id
    JOIN movie_keyword AS mk ON mk.movie_id = t.id
    JOIN keyword AS k ON k.id = mk.keyword_id
    WHERE t.production_year BETWEEN 2000 AND 2010
      AND t.kind_id IN (1, 3, 4)
      AND ((((t.id % 997))::smallint - (length(t.title))::smallint) < GCD(((t.production_year - 1900))::integer, (t.kind_id)::smallint)) AND (BTRIM(BTRIM((COALESCE(cn.name, ''))::text, (COALESCE(cn.country_code, ''))::text), BTRIM((k.keyword)::text, (COALESCE(k.phonetic_code, ''))::text)) < BTRIM(BTRIM((t.title)::text, (kt.kind)::text), BTRIM((ct.kind)::text, (COALESCE(cn.name_pcode_nf, ''))::text)))
    GROUP BY t.production_year, kt.kind, cn.country_code, ct.kind
)
SELECT production_year, movie_kind, country_code, company_role, movies,
       keyword_count, company_keyword_mentions,
       dense_rank() OVER (
           PARTITION BY production_year, movie_kind ORDER BY movies DESC
       ) AS country_company_rank
FROM coverage;
