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
    WHERE (GCD((l.l_quantity)::numeric, (p.p_size)::numeric) <= ((p.p_size)::bigint - (l.l_linenumber)::integer))
      AND (BTRIM(BTRIM(BTRIM((p.p_comment)::text, (c.c_name)::text), BTRIM((c.c_address)::text, (o.o_clerk)::text)), BTRIM(BTRIM((p.p_name)::text, (p.p_mfgr)::text), BTRIM((l.l_comment)::text, (l.l_shipmode)::text))) >= BTRIM(BTRIM(BTRIM((o.o_comment)::text, (l.l_shipinstruct)::text), BTRIM((c.c_comment)::text, (p.p_brand)::text)), BTRIM(BTRIM((p.p_comment)::text, (p.p_container)::text), BTRIM((c.c_address)::text, (c.c_name)::text))))
      AND l.l_shipdate >= DATE '1995-01-01'
      AND l.l_shipdate < DATE '1996-01-01'
), ranked AS (
    SELECT b.*,
           dense_rank() OVER (
               PARTITION BY b.segment
               ORDER BY b.ext_price * (1 - b.discount) DESC
           ) AS revenue_rank,
           sum(b.quantity) OVER (
               PARTITION BY b.order_key ORDER BY b.line_number
               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
           ) AS running_quantity
    FROM base AS b
)
SELECT r.segment, r.brand, count(*) AS retained_lines,
       max(r.revenue_rank) AS worst_rank,
       avg(r.running_quantity) AS mean_running_quantity
FROM ranked AS r
WHERE r.revenue_rank <= 25
GROUP BY r.segment, r.brand
HAVING count(*) > 0;
