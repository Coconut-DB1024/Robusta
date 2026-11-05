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
    WHERE (GCD((p.p_size)::integer, (l.l_quantity)::smallint) <= ((l.l_linenumber)::smallint * (l.l_linenumber)::smallint))
      AND (BTRIM(BTRIM(BTRIM((o.o_comment)::text, (l.l_shipmode)::text), BTRIM((c.c_comment)::text, (l.l_shipinstruct)::text)), BTRIM(BTRIM((p.p_comment)::text, (p.p_brand)::text), BTRIM((c.c_address)::text, (p.p_container)::text))) NOT IN (BTRIM(BTRIM(BTRIM((p.p_name)::text, (c.c_name)::text), BTRIM((l.l_comment)::text, (o.o_clerk)::text)), BTRIM(BTRIM((o.o_comment)::text, (p.p_mfgr)::text), BTRIM((c.c_comment)::text, (l.l_shipmode)::text)))))
      AND l.l_shipdate >= DATE '1995-01-01'
      AND l.l_shipdate < DATE '1996-01-01'
), shipped AS (
    SELECT b.order_key, b.segment,
           count(*) AS shipped_lines,
           sum(b.ext_price * (1 - b.discount)) AS shipped_revenue
    FROM base AS b
    GROUP BY b.order_key, b.segment
), received AS (
    SELECT li.l_orderkey AS order_key,
           count(*) AS received_lines,
           sum(li.l_extendedprice) AS gross_received_value
    FROM lineitem AS li
    WHERE li.l_receiptdate >= DATE '1995-01-01'
      AND li.l_receiptdate < DATE '1996-01-01'
    GROUP BY li.l_orderkey
)
SELECT coalesce(s.segment, 'unknown') AS segment,
       count(*) AS orders,
       sum(coalesce(s.shipped_lines, 0)) AS shipped_lines,
       sum(coalesce(rc.received_lines, 0)) AS received_lines,
       sum(coalesce(s.shipped_revenue, 0)) AS net_revenue,
       sum(coalesce(rc.gross_received_value, 0)) AS gross_value
FROM shipped AS s
FULL OUTER JOIN received AS rc ON rc.order_key = s.order_key
GROUP BY coalesce(s.segment, 'unknown')
HAVING count(*) > 0;
