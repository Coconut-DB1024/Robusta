WITH eligible_orders AS (
 SELECT o.o_orderkey,o.o_custkey,o.o_orderpriority,o.o_totalprice
 FROM orders o
 WHERE o.o_orderdate>=DATE '1993-01-01' AND o.o_orderdate<DATE '1997-01-01'
 AND EXISTS (SELECT 1 FROM lineitem lx WHERE lx.l_orderkey=o.o_orderkey AND lx.l_returnflag='R')
)
SELECT c.c_mktsegment,e.o_orderpriority,p.p_brand,
 count(DISTINCT e.o_orderkey) AS retained_orders,
 sum(l.l_extendedprice*(1-l.l_discount)) AS revenue,
 avg(e.o_totalprice) AS mean_order_total
FROM eligible_orders e JOIN customer c ON c.c_custkey=e.o_custkey
JOIN lineitem l ON l.l_orderkey=e.o_orderkey JOIN part p ON p.p_partkey=l.l_partkey
WHERE ((((CAST((l.l_quantity) AS smallint) # CAST((p.p_size) AS smallint)) # (CAST((l.l_linenumber) AS smallint) # CAST((l.l_quantity) AS smallint))) > ((CAST((p.p_size) AS smallint) # CAST((l.l_linenumber) AS smallint)) # (CAST((l.l_quantity) AS smallint) # CAST((p.p_size) AS smallint)))))
AND l.l_shipdate>l.l_commitdate AND l.l_receiptdate>l.l_shipdate
AND c.c_acctbal>0 AND p.p_size BETWEEN 10 AND 45
GROUP BY c.c_mktsegment,e.o_orderpriority,p.p_brand
HAVING count(*)>3;
