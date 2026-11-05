WITH eligible_orders AS (
 SELECT o.o_orderkey,o.o_custkey,o.o_orderpriority,o.o_totalprice
 FROM orders o
 WHERE o.o_orderdate>=DATE '1992-01-01' AND o.o_orderdate<DATE '1998-01-01'
 AND EXISTS (SELECT 1 FROM lineitem lx WHERE lx.l_orderkey=o.o_orderkey AND lx.l_returnflag='R')
)
SELECT n.n_name,e.o_orderpriority,p.p_brand,
 count(DISTINCT e.o_orderkey) AS returned_orders,
 count(DISTINCT e.o_custkey) AS buyers,
 sum(l.l_extendedprice*(1-l.l_discount)-l.l_quantity*ps.ps_supplycost) AS delayed_margin,
 sum(l.l_quantity) AS delayed_units,
 count(*) FILTER (WHERE ps.ps_availqty<1000) AS low_stock_lines
FROM eligible_orders e JOIN customer c ON c.c_custkey=e.o_custkey
JOIN lineitem l ON l.l_orderkey=e.o_orderkey JOIN part p ON p.p_partkey=l.l_partkey
JOIN partsupp ps ON ps.ps_partkey=l.l_partkey AND ps.ps_suppkey=l.l_suppkey
JOIN supplier s ON s.s_suppkey=ps.ps_suppkey JOIN nation n ON n.n_nationkey=s.s_nationkey
WHERE ((((CAST((l.l_quantity) AS smallint) # CAST((p.p_size) AS smallint)) # (CAST((l.l_linenumber) AS smallint) # CAST((l.l_quantity) AS smallint))) > ((CAST((p.p_size) AS smallint) # CAST((l.l_linenumber) AS smallint)) # (CAST((l.l_quantity) AS smallint) # CAST((p.p_size) AS smallint)))))
AND l.l_shipdate>l.l_commitdate AND l.l_receiptdate>l.l_shipdate
AND c.c_acctbal>0 AND p.p_size BETWEEN 10 AND 45 AND ps.ps_availqty>l.l_quantity
GROUP BY n.n_name,e.o_orderpriority,p.p_brand
HAVING count(*)>3;
