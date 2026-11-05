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
    WHERE (((l.l_quantity)::smallint - (p.p_size)::bigint) != CEILING((l.l_linenumber)::integer))
      AND ((l.l_comment)::text BETWEEN BTRIM(BTRIM(BTRIM((c.c_address)::text, (c.c_name)::text), BTRIM((p.p_name)::text, (o.o_clerk)::text)), BTRIM(BTRIM((l.l_comment)::text, (p.p_mfgr)::text), BTRIM((o.o_comment)::text, (l.l_shipmode)::text))) AND BTRIM(BTRIM(BTRIM((c.c_comment)::text, (l.l_shipinstruct)::text), BTRIM((p.p_comment)::text, (p.p_brand)::text)), BTRIM(BTRIM((c.c_address)::text, (p.p_container)::text), BTRIM((p.p_name)::text, (c.c_name)::text))))
      AND l.l_shipdate >= DATE '1995-01-01'
      AND l.l_shipdate < DATE '1996-01-01'
), pairs AS (
    SELECT a.order_key, a.segment, a.brand AS first_brand,
           z.brand AS second_brand, a.quantity + z.quantity AS pair_quantity,
           a.ext_price * (1 - a.discount) + z.ext_price * (1 - z.discount) AS pair_revenue
    FROM base AS a
    JOIN base AS z
      ON z.order_key = a.order_key
     AND z.line_number > a.line_number
)
SELECT p.segment, p.first_brand, p.second_brand,
       count(*) AS line_pairs, avg(p.pair_quantity) AS mean_pair_quantity,
       sum(p.pair_revenue) AS pair_revenue
FROM pairs AS p
GROUP BY p.segment, p.first_brand, p.second_brand
HAVING count(*) > 0;
