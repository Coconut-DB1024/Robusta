WITH eligible_orders AS (
 SELECT o.o_orderkey,o.o_custkey,o.o_orderpriority,o.o_totalprice
 FROM orders o
 WHERE o.o_orderdate>=DATE '1992-01-01' AND o.o_orderdate<DATE '1998-01-01'
 AND EXISTS (SELECT 1 FROM lineitem lx WHERE lx.l_orderkey=o.o_orderkey AND lx.l_returnflag='R')
)
SELECT c.c_mktsegment,c.c_nationkey,e.o_orderpriority,p.p_brand,l.l_shipmode,
 count(DISTINCT e.o_orderkey) AS returned_orders,
 sum(l.l_extendedprice*(1-l.l_discount)) AS delayed_shipping_revenue,
 avg(e.o_totalprice) AS mean_returned_order_total,
 sum(l.l_quantity) AS shipped_quantity
FROM eligible_orders e JOIN customer c ON c.c_custkey=e.o_custkey
JOIN lineitem l ON l.l_orderkey=e.o_orderkey JOIN part p ON p.p_partkey=l.l_partkey
WHERE (((36.78 * CAST((l.l_quantity) AS real)) - (CAST((p.p_size) AS bigint) + 276)) < ((CAST((l.l_quantity) AS integer) * CAST((p.p_size) AS bigint)) - (CAST((p.p_size) AS double precision) - CAST((l.l_quantity) AS smallint)))) AND l.l_shipdate>l.l_commitdate AND l.l_receiptdate>l.l_shipdate
AND c.c_acctbal>0 AND p.p_size BETWEEN 10 AND 45
GROUP BY c.c_mktsegment,c.c_nationkey,e.o_orderpriority,p.p_brand,l.l_shipmode
HAVING count(*)>1;
