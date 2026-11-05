WITH markets AS (
    SELECT c.c_mktsegment AS segment, sn.n_name AS supplier_nation,
           cn.n_name AS customer_nation,
           count(DISTINCT o.o_orderkey) AS order_count,
           sum(l.l_extendedprice * (1 - l.l_discount)) AS revenue,
           count(*) FILTER (WHERE l.l_receiptdate > l.l_commitdate) AS late_lines
    FROM lineitem AS l
    JOIN orders AS o ON o.o_orderkey = l.l_orderkey
    JOIN part AS p ON p.p_partkey = l.l_partkey
    JOIN customer AS c ON c.c_custkey = o.o_custkey
    JOIN supplier AS s ON s.s_suppkey = l.l_suppkey
    JOIN nation AS sn ON sn.n_nationkey = s.s_nationkey
    JOIN nation AS cn ON cn.n_nationkey = c.c_nationkey
    WHERE l.l_shipdate >= DATE '1995-01-01'
      AND l.l_shipdate < DATE '1996-01-01'
      AND (((l.l_quantity)::smallint - (p.p_size)::smallint) < GCD((p.p_size)::integer, (l.l_linenumber)::smallint)) AND (BTRIM(BTRIM((l.l_comment)::text, (p.p_brand)::text), BTRIM((o.o_comment)::text, (p.p_container)::text)) < BTRIM(BTRIM((c.c_comment)::text, (c.c_name)::text), BTRIM((p.p_name)::text, (o.o_clerk)::text)))
    GROUP BY c.c_mktsegment, sn.n_name, cn.n_name
)
SELECT segment, supplier_nation, customer_nation, order_count, revenue, late_lines,
       dense_rank() OVER (PARTITION BY segment ORDER BY revenue DESC) AS market_rank
FROM markets
ORDER BY segment, market_rank, supplier_nation, customer_nation;
