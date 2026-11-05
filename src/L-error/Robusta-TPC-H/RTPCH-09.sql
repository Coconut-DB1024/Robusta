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
    WHERE (OCTET_LENGTH('admin') >= ((p.p_size)::smallint # (l.l_linenumber)::smallint))
      AND (BTRIM(BTRIM(BTRIM((c.c_address)::text, (p.p_mfgr)::text), BTRIM((p.p_name)::text, (l.l_shipmode)::text)), BTRIM(BTRIM((l.l_comment)::text, (l.l_shipinstruct)::text), BTRIM((o.o_comment)::text, (p.p_brand)::text))) IS DISTINCT FROM BTRIM(BTRIM(BTRIM((c.c_comment)::text, (p.p_container)::text), BTRIM((p.p_comment)::text, (c.c_name)::text)), BTRIM(BTRIM((c.c_address)::text, (o.o_clerk)::text), BTRIM((p.p_name)::text, (p.p_mfgr)::text))))
      AND l.l_shipdate >= DATE '1995-01-01'
      AND l.l_shipdate < DATE '1996-01-01'
), supply_cost AS (
    SELECT ps.ps_partkey, ps.ps_suppkey,
           sum(ps.ps_supplycost * ps.ps_availqty) AS inventory_value,
           count(*) AS stock_rows
    FROM partsupp AS ps
    GROUP BY ps.ps_partkey, ps.ps_suppkey
), activity AS (
    SELECT b.brand, b.part_type, b.part_key, b.supplier_key,
           sum(b.ext_price * (1 - b.discount)) AS revenue,
           count(DISTINCT b.order_key) AS orders
    FROM base AS b
    GROUP BY b.brand, b.part_type, b.part_key, b.supplier_key
)
SELECT a.brand, a.part_type, count(*) AS active_pairs,
       sum(a.revenue) AS revenue, sum(sc.inventory_value) AS inventory_value
FROM activity AS a
JOIN supply_cost AS sc
  ON sc.ps_partkey = a.part_key AND sc.ps_suppkey = a.supplier_key
GROUP BY a.brand, a.part_type
HAVING sum(a.orders) > 0;
