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
    WHERE (((l.l_quantity)::smallint - (p.p_size)::smallint) > GCD((p.p_size)::integer, (l.l_linenumber)::smallint))
      AND ((o.o_comment)::text BETWEEN BTRIM(BTRIM((p.p_name)::text, (p.p_mfgr)::text), BTRIM((l.l_comment)::text, (l.l_shipmode)::text)) AND BTRIM(BTRIM((o.o_comment)::text, (l.l_shipinstruct)::text), BTRIM((c.c_comment)::text, (p.p_brand)::text)))
      AND l.l_shipdate >= DATE '1995-01-01'
      AND l.l_shipdate < DATE '1996-01-01'
), sampled_orders AS (
    SELECT DISTINCT b.order_key, b.part_key, b.segment, b.brand
    FROM base AS b
)
SELECT so.segment, so.brand, count(*) AS order_part_pairs,
       sum(x.line_count) AS matching_lines,
       avg(x.mean_quantity) AS mean_quantity
FROM sampled_orders AS so
CROSS JOIN LATERAL (
    SELECT count(*) AS line_count, avg(li.l_quantity) AS mean_quantity
    FROM lineitem AS li
    WHERE li.l_orderkey = so.order_key
      AND li.l_partkey = so.part_key
) AS x
GROUP BY so.segment, so.brand
HAVING sum(x.line_count) > 0;
