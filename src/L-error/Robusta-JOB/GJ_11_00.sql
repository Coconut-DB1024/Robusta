-- Kalita-JOB feedback-guided: GJOB_T11; variation 0; initial
WITH credits AS (
    SELECT t.id AS movie_id, t.production_year, rt.role,
           count(DISTINCT n.id) AS people
    FROM title t JOIN cast_info ci ON ci.movie_id = t.id
    JOIN name n ON n.id = ci.person_id JOIN role_type rt ON rt.id = ci.role_id
    WHERE t.production_year BETWEEN 2006 AND 2010 AND t.kind_id = 1
      AND ((CAST(((t.production_year - 1900)) AS numeric) != LENGTH(CAST((n.name) AS text)))) AND ((CHAR_LENGTH(UPPER(CAST((t.title) AS text))) != CAST((ci.nr_order) AS numeric)))
    GROUP BY t.id, t.production_year, rt.role
), cohorts AS (
    SELECT c.role, c.production_year, subject.kind AS subject_kind,
           status.kind AS status_kind, count(DISTINCT c.movie_id) AS movies,
           sum(c.people) AS people_sum
    FROM credits c JOIN complete_cast cc ON cc.movie_id = c.movie_id
    JOIN comp_cast_type subject ON subject.id = cc.subject_id
    JOIN comp_cast_type status ON status.id = cc.status_id
    WHERE EXISTS (SELECT 1 FROM movie_keyword mk
                  JOIN keyword k ON k.id = mk.keyword_id WHERE mk.movie_id = c.movie_id)
    GROUP BY c.role, c.production_year, subject.kind, status.kind
)
SELECT *, avg(people_sum) OVER (PARTITION BY role, subject_kind) AS cohort_mean_people,
       dense_rank() OVER (PARTITION BY role ORDER BY movies DESC) AS coverage_rank
FROM cohorts
