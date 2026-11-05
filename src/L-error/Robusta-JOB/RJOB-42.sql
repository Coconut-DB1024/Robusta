SELECT t.kind_id, left_keyword.phonetic_code AS keyword_family,
       count(*) AS cooccurrences, count(DISTINCT t.id) AS movies,
       count(DISTINCT right_keyword.id) AS companion_keywords
FROM title AS t
JOIN movie_keyword AS left_tag ON left_tag.movie_id = t.id
JOIN keyword AS left_keyword ON left_keyword.id = left_tag.keyword_id
JOIN movie_keyword AS right_tag
  ON right_tag.movie_id = t.id AND right_tag.keyword_id > left_tag.keyword_id
JOIN keyword AS right_keyword ON right_keyword.id = right_tag.keyword_id
WHERE t.production_year BETWEEN 2005 AND 2010
  AND t.kind_id = 1 AND (((CAST(((t.id % 997)) AS numeric) - CAST((length(t.title)) AS smallint)) <= OCTET_LENGTH('test_data'))) AND ((BTRIM(BTRIM(BTRIM(CAST((COALESCE(left_keyword.keyword, '')) AS text), CAST((COALESCE(left_keyword.phonetic_code, '')) AS text)), BTRIM(CAST((COALESCE(right_keyword.keyword, '')) AS text), CAST((COALESCE(right_keyword.phonetic_code, '')) AS text))), BTRIM(BTRIM(CAST((COALESCE(right_keyword.keyword, '')) AS text), CAST((COALESCE(right_keyword.phonetic_code, '')) AS text)), BTRIM(CAST((COALESCE(left_keyword.phonetic_code, '')) AS text), CAST((COALESCE(left_keyword.keyword, '')) AS text)))) != BTRIM(BTRIM(BTRIM(CAST((COALESCE(left_keyword.phonetic_code, '')) AS text), CAST((COALESCE(right_keyword.phonetic_code, '')) AS text)), BTRIM(CAST((COALESCE(left_keyword.keyword, '')) AS text), CAST((COALESCE(right_keyword.keyword, '')) AS text))), BTRIM(BTRIM(CAST((COALESCE(right_keyword.phonetic_code, '')) AS text), CAST((COALESCE(left_keyword.keyword, '')) AS text)), BTRIM(CAST((COALESCE(right_keyword.keyword, '')) AS text), CAST((COALESCE(left_keyword.phonetic_code, '')) AS text))))))
GROUP BY t.kind_id, left_keyword.phonetic_code
HAVING count(DISTINCT right_keyword.id) >= 1
