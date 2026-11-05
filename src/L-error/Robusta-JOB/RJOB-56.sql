-- Guided v4 V4_EPISODES_01: Long-term episode-level actor and named-character coverage for series with alternative titles. Episode-to-series and alternative-title relationships both determine membership.
SELECT series.id AS series_id,series.title AS series_title,episode.production_year,
 rt.role,n.gender,count(DISTINCT episode.id) AS episodes,
 count(DISTINCT n.id) AS credited_people,count(DISTINCT ch.id) AS named_characters
FROM title episode JOIN title series ON series.id=episode.episode_of_id
JOIN cast_info ci ON ci.movie_id=episode.id
JOIN name n ON n.id=ci.person_id JOIN char_name ch ON ch.id=ci.person_role_id
JOIN role_type rt ON rt.id=ci.role_id
WHERE episode.production_year BETWEEN 1990 AND 2012
AND episode.kind_id=7 AND series.kind_id=2 AND series.production_year BETWEEN 1990 AND 2007
AND ci.role_id IN (1,2) AND (CAST((length(series.title)) AS integer) < CAST((length(episode.title)) AS integer))
AND EXISTS (SELECT 1 FROM aka_title atitle
 WHERE atitle.movie_id=series.id AND atitle.title<>series.title)
GROUP BY series.id,series.title,episode.production_year,rt.role,n.gender;
