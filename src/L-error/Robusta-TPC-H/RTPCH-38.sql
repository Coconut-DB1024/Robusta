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
    WHERE (((l.l_quantity)::numeric - (p.p_size)::numeric) NOT IN (GCD((l.l_quantity)::numeric, (p.p_size)::numeric)))
      AND (BTRIM(BTRIM((p.p_name)::text, (c.c_name)::text), BTRIM((l.l_comment)::text, (o.o_clerk)::text)) IS DISTINCT FROM BTRIM(BTRIM((o.o_comment)::text, (p.p_mfgr)::text), BTRIM((c.c_comment)::text, (l.l_shipmode)::text)))
      AND l.l_shipdate >= DATE '1995-01-01'
      AND l.l_shipdate < DATE '1996-01-01'
), part_activity AS (
    SELECT b.part_key, b.brand, b.part_type, b.segment,
           count(DISTINCT b.order_key) AS orders,
           sum(b.quantity) AS quantity,
           sum(b.ext_price * (1 - b.discount)) AS revenue
    FROM base AS b
    WHERE b.order_status IN ('F', 'O', 'P')
    GROUP BY b.part_key, b.brand, b.part_type, b.segment
), qualified AS (
    SELECT pa.*,
           sum(pa.revenue) OVER (PARTITION BY pa.brand, pa.segment) AS brand_revenue,
           row_number() OVER (
               PARTITION BY pa.segment ORDER BY pa.quantity DESC, pa.part_key
           ) AS segment_rank
    FROM part_activity AS pa
    WHERE EXISTS (
        SELECT 1 FROM partsupp AS ps
        JOIN supplier AS s ON s.s_suppkey = ps.ps_suppkey
        JOIN nation AS n ON n.n_nationkey = s.s_nationkey
        WHERE ps.ps_partkey = pa.part_key
          AND ps.ps_availqty > 0
          AND n.n_regionkey BETWEEN 0 AND 4
    )
)
SELECT q.segment, q.brand, count(*) AS qualified_parts,
       sum(q.orders) AS orders, sum(q.quantity) AS quantity,
       max(q.brand_revenue) AS brand_revenue,
       min(q.segment_rank) AS best_segment_rank
FROM qualified AS q
WHERE q.segment_rank <= 100
GROUP BY q.segment, q.brand
HAVING sum(q.revenue) > 0;
