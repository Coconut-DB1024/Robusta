SELECT t.production_year, t.kind_id, count(*) AS movie_count
FROM title AS t
WHERE t.production_year BETWEEN 1998 AND 2010
  AND t.kind_id IN (1, 3, 4) AND (((CAST(((t.id % 997)) AS smallint) - CAST((length(t.title)) AS smallint)) > GCD(CAST(((t.production_year - 1900)) AS integer), CAST((t.kind_id) AS smallint))))
  AND t.id IN (
      SELECT mc.movie_id
      FROM movie_companies AS mc
      JOIN company_name AS cn ON cn.id = mc.company_id
      WHERE ((BTRIM(CAST((COALESCE(cn.name, '')) AS text), CAST((COALESCE(cn.country_code, '')) AS text)) NOT IN (BTRIM(CAST((COALESCE(cn.name_pcode_nf, '')) AS text), CAST((COALESCE(cn.name_pcode_sf, '')) AS text))))) AND cn.country_code IS NOT NULL
  )
  AND EXISTS (
      SELECT 1 FROM movie_keyword AS mk
      JOIN keyword AS k ON k.id = mk.keyword_id
      WHERE mk.movie_id = t.id AND length(k.keyword) > 3
  )
  AND EXISTS (
      SELECT 1 FROM movie_info_idx AS mii WHERE mii.movie_id = t.id
  )
GROUP BY t.production_year, t.kind_id
