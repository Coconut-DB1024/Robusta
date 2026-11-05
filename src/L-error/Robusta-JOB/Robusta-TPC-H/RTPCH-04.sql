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
    WHERE (FLOOR((l.l_quantity)::integer) IS DISTINCT FROM ((p.p_size)::smallint # (l.l_linenumber)::smallint))
      AND (BTRIM(BTRIM((c.c_address)::text, (p.p_mfgr)::text), BTRIM((p.p_name)::text, (l.l_shipmode)::text)) NOT IN (BTRIM(BTRIM((l.l_comment)::text, (l.l_shipinstruct)::text), BTRIM((o.o_comment)::text, (p.p_brand)::text))))
      AND l.l_shipdate >= DATE '1995-01-01'
      AND l.l_shipdate < DATE '1996-01-01'
), eligible AS (
    SELECT b.part_key, b.brand, b.part_type,
           sum(b.ext_price * (1 - b.discount)) AS sales,
           count(DISTINCT b.order_key) AS order_count
    FROM base AS b
    WHERE EXISTS (
        SELECT 1
        FROM partsupp AS ps
        JOIN supplier AS s ON s.s_suppkey = ps.ps_suppkey
        JOIN nation AS n ON n.n_nationkey = s.s_nationkey
        WHERE ps.ps_partkey = b.part_key
          AND ps.ps_suppkey = b.supplier_key
          AND ps.ps_availqty > 0
          AND n.n_regionkey BETWEEN 0 AND 4
    )
    GROUP BY b.part_key, b.brand, b.part_type
)
SELECT e.brand, e.part_type, count(*) AS qualifying_parts,
       sum(e.sales) AS sales, avg(e.order_count) AS mean_orders
FROM eligible AS e
GROUP BY e.brand, e.part_type
HAVING sum(e.sales) > 0;
