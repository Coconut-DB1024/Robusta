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
    WHERE (GCD((l.l_quantity)::numeric, (p.p_size)::numeric) < ((p.p_size)::bigint - (l.l_linenumber)::integer))
      AND (BTRIM(BTRIM(BTRIM((l.l_comment)::text, (p.p_mfgr)::text), BTRIM((o.o_comment)::text, (l.l_shipmode)::text)), BTRIM(BTRIM((c.c_comment)::text, (l.l_shipinstruct)::text), BTRIM((p.p_comment)::text, (p.p_brand)::text))) < BTRIM(BTRIM(BTRIM((c.c_address)::text, (p.p_container)::text), BTRIM((p.p_name)::text, (c.c_name)::text)), BTRIM(BTRIM((l.l_comment)::text, (o.o_clerk)::text), BTRIM((o.o_comment)::text, (p.p_mfgr)::text))))
      AND l.l_shipdate >= DATE '1995-01-01'
      AND l.l_shipdate < DATE '1996-01-01'
), supplier_part AS (
    SELECT b.supplier_key, b.part_key, b.brand, b.segment,
           sum(b.quantity) AS shipped_quantity,
           sum(b.ext_price * (1 - b.discount)) AS revenue
    FROM base AS b
    GROUP BY b.supplier_key, b.part_key, b.brand, b.segment
    HAVING sum(b.quantity) > 0
), scored AS (
    SELECT sp.*, ps.ps_availqty, ps.ps_supplycost,
           sp.revenue - ps.ps_supplycost * sp.shipped_quantity AS margin_proxy
    FROM supplier_part AS sp
    JOIN partsupp AS ps
      ON ps.ps_partkey = sp.part_key AND ps.ps_suppkey = sp.supplier_key
)
SELECT s.brand, s.segment, count(*) AS supplier_parts,
       sum(s.margin_proxy) AS total_margin_proxy,
       avg(s.ps_availqty) AS mean_availability
FROM scored AS s
WHERE (s.supplier_key, s.part_key) IN (
    SELECT b.supplier_key, b.part_key
    FROM base AS b
    GROUP BY b.supplier_key, b.part_key
    HAVING count(DISTINCT b.customer_key) > 1
)
GROUP BY s.brand, s.segment
HAVING sum(s.margin_proxy) IS NOT NULL;
