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
    WHERE (((l.l_quantity)::smallint - (p.p_size)::smallint) < GCD((p.p_size)::integer, (l.l_linenumber)::smallint))
      AND (BTRIM((p.p_name)::text, (o.o_clerk)::text) > BTRIM((l.l_comment)::text, (p.p_mfgr)::text))
      AND l.l_shipdate >= DATE '1995-01-01'
      AND l.l_shipdate < DATE '1996-01-01'
), customer_month AS (
    SELECT b.customer_key, b.nation_key,
           date_trunc('month', b.order_date)::date AS order_month,
           sum(b.ext_price * (1 - b.discount)) AS revenue,
           count(DISTINCT b.order_key) AS orders
    FROM base AS b
    GROUP BY b.customer_key, b.nation_key,
             date_trunc('month', b.order_date)::date
), nation_month AS (
    SELECT cm.nation_key, cm.order_month, sum(cm.revenue) AS revenue,
           sum(cm.orders) AS orders, count(*) AS customer_months
    FROM customer_month AS cm
    GROUP BY cm.nation_key, cm.order_month
)
SELECT n.n_name, r.r_name, nm.order_month, nm.revenue,
       sum(nm.revenue) OVER (
           PARTITION BY n.n_regionkey ORDER BY nm.order_month
           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
       ) AS regional_running_revenue,
       nm.orders, nm.customer_months
FROM nation_month AS nm
JOIN nation AS n ON n.n_nationkey = nm.nation_key
JOIN region AS r ON r.r_regionkey = n.n_regionkey;
