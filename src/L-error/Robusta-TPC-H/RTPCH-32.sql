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
    WHERE (GCD((l.l_quantity)::bigint, 1) BETWEEN ((l.l_linenumber)::smallint # (o.o_shippriority)::smallint) AND ((p.p_size)::smallint # (l.l_quantity)::smallint))
      AND (BTRIM(BTRIM(BTRIM((c.c_comment)::text, (l.l_shipinstruct)::text), BTRIM((p.p_comment)::text, (p.p_brand)::text)), BTRIM(BTRIM((c.c_address)::text, (p.p_container)::text), BTRIM((p.p_name)::text, (c.c_name)::text))) > BTRIM(BTRIM(BTRIM((l.l_comment)::text, (o.o_clerk)::text), BTRIM((o.o_comment)::text, (p.p_mfgr)::text)), BTRIM(BTRIM((c.c_comment)::text, (l.l_shipmode)::text), BTRIM((p.p_comment)::text, (l.l_shipinstruct)::text))))
      AND l.l_shipdate >= DATE '1995-01-01'
      AND l.l_shipdate < DATE '1996-01-01'
), by_order AS (
    SELECT b.order_key, b.segment, b.brand,
           sum(b.ext_price * (1 - b.discount)) AS revenue
    FROM base AS b
    GROUP BY b.order_key, b.segment, b.brand
), receipts AS (
    SELECT li.l_orderkey AS order_key,
           count(*) FILTER (WHERE li.l_receiptdate >= li.l_commitdate) AS on_or_late,
           max(li.l_receiptdate - li.l_shipdate) AS max_transit_days
    FROM lineitem AS li
    WHERE li.l_shipdate >= DATE '1995-01-01'
      AND li.l_shipdate < DATE '1996-01-01'
    GROUP BY li.l_orderkey
)
SELECT bo.segment, bo.brand, count(*) AS orders,
       sum(bo.revenue) AS revenue,
       sum(coalesce(rc.on_or_late, 0)) AS on_or_late_lines,
       max(rc.max_transit_days) AS max_transit_days
FROM by_order AS bo
LEFT JOIN receipts AS rc ON rc.order_key = bo.order_key
GROUP BY bo.segment, bo.brand
HAVING count(*) > 0;
