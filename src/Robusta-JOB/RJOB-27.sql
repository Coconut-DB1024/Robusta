WITH information_counts AS (
    SELECT mii.movie_id, mii.info_type_id, count(*) AS information_rows
    FROM movie_info_idx AS mii
    JOIN info_type AS source_info ON source_info.id = mii.info_type_id
    JOIN title AS source_title ON source_title.id = mii.movie_id
    WHERE (((BTRIM(CAST((COALESCE(mii.info, '')) AS text), CAST((COALESCE(mii.note, '')) AS text)) != BTRIM(CAST((COALESCE(source_info.info, '')) AS text), CAST((COALESCE(source_title.title, '')) AS text))) OR (BTRIM(CAST((COALESCE(source_info.info, '')) AS text), CAST((COALESCE(source_title.title, '')) AS text)) != BTRIM(CAST((COALESCE(mii.note, '')) AS text), CAST((COALESCE(mii.info, '')) AS text)))))
    GROUP BY mii.movie_id, mii.info_type_id
)
SELECT it.info, t.production_year,
       count(*) AS movie_information_groups,
       sum(ic.information_rows) AS source_rows
FROM title AS t
JOIN information_counts AS ic ON ic.movie_id = t.id
JOIN info_type AS it ON it.id = ic.info_type_id
WHERE t.production_year BETWEEN 1995 AND 2010
  AND t.kind_id = 1 AND ((GCD(1, CAST((t.kind_id) AS integer)) < (390 * CAST(((length(t.title) - (t.id % 31))) AS double precision))))
GROUP BY it.info, t.production_year
HAVING sum(ic.information_rows) >= count(*)
