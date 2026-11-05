WITH base AS (
    SELECT l.l_orderkey AS order_key,
           l.l_partkey AS part_key,
           l.l_suppkey AS supplier_key,
           l.l_linenumber AS line_number,
           l.l_quantity AS quantity,
           l.l_extendedprice AS ext_price,
           l.l_discount AS discount,
           l.l_tax AS tax,
           l.l_shipdate AS ship_date,
           l.l_commitdate AS commit_date,
           l.l_receiptdate AS receipt_date,
           l.l_shipmode AS ship_mode,
           l.l_returnflag AS return_flag,
           o.o_custkey AS customer_key,
           o.o_orderdate AS order_date,
           o.o_orderstatus AS order_status,
           o.o_orderpriority AS priority,
           o.o_totalprice AS order_total,
           o.o_shippriority AS ship_priority,
           c.c_name AS customer_name,
           c.c_mktsegment AS segment,
           c.c_nationkey AS nation_key,
           c.c_acctbal AS acct_balance,
           p.p_name AS part_name,
           p.p_brand AS brand,
           p.p_type AS part_type,
           p.p_size AS part_size,
           p.p_retailprice AS retail_price,
           l.l_comment AS line_comment,
           o.o_comment AS order_comment
    FROM lineitem AS l
    JOIN orders AS o ON o.o_orderkey = l.l_orderkey
    JOIN part AS p ON p.p_partkey = l.l_partkey
    JOIN customer AS c ON c.c_custkey = o.o_custkey
    WHERE (GCD((l.l_quantity)::smallint, 1) <= ((p.p_size)::integer # (l.l_linenumber)::integer))
      AND ((c.c_address)::text BETWEEN SYMMETRIC BTRIM(BTRIM((c.c_comment)::text, (c.c_name)::text), BTRIM((p.p_comment)::text, (o.o_clerk)::text)) AND BTRIM(BTRIM((c.c_address)::text, (p.p_mfgr)::text), BTRIM((p.p_name)::text, (l.l_shipmode)::text)))
      AND l.l_shipdate >= DATE '1995-01-01'
      AND l.l_shipdate < DATE '1996-01-01'
), customer_baseline AS (
    SELECT o2.o_custkey AS customer_key, avg(o2.o_totalprice) AS avg_total
    FROM orders AS o2
    GROUP BY o2.o_custkey
), order_rollup AS (
    SELECT b.order_key, b.customer_key, b.priority, b.segment,
           sum(b.ext_price * (1 - b.discount)) AS line_revenue,
           count(*) AS line_count
    FROM base AS b
    JOIN customer_baseline AS cb ON cb.customer_key = b.customer_key
    GROUP BY b.order_key, b.customer_key, b.priority, b.segment, cb.avg_total
    HAVING sum(b.ext_price) > cb.avg_total
)
SELECT g.segment, g.priority, count(*) AS qualifying_orders,
       sum(g.line_revenue) AS revenue, avg(g.line_count) AS mean_lines
FROM order_rollup AS g
GROUP BY g.segment, g.priority;
