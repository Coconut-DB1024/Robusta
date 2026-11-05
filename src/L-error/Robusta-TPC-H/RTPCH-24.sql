WITH inventory AS (
    SELECT ps.ps_suppkey AS supplier_key, p.p_brand AS brand,
           sum(ps.ps_availqty * ps.ps_supplycost) AS inventory_value
    FROM partsupp AS ps JOIN part AS p ON p.p_partkey = ps.ps_partkey
    GROUP BY ps.ps_suppkey, p.p_brand
), sales AS (
    SELECT s.s_suppkey AS supplier_key, n.n_name AS nation, p.p_brand AS brand,
           sum(l.l_extendedprice * (1 - l.l_discount)) AS revenue,
           count(DISTINCT l.l_orderkey) AS order_count
    FROM lineitem AS l
    JOIN partsupp AS ps ON ps.ps_partkey = l.l_partkey AND ps.ps_suppkey = l.l_suppkey
    JOIN part AS p ON p.p_partkey = ps.ps_partkey
    JOIN supplier AS s ON s.s_suppkey = ps.ps_suppkey
    JOIN nation AS n ON n.n_nationkey = s.s_nationkey
    WHERE l.l_shipdate >= DATE '1995-01-01'
      AND l.l_shipdate < DATE '1996-01-01'
      AND (GCD((l.l_quantity)::numeric, (p.p_size)::numeric) >= ((p.p_size)::bigint - (l.l_linenumber)::integer)) AND ((l.l_comment)::text BETWEEN SYMMETRIC BTRIM((s.s_comment)::text, (p.p_brand)::text) AND BTRIM((p.p_comment)::text, (p.p_container)::text))
    GROUP BY s.s_suppkey, n.n_name, p.p_brand
)
SELECT sales.nation, sales.brand, count(*) AS active_suppliers,
       sum(sales.order_count) AS supplier_orders, sum(sales.revenue) AS revenue,
       sum(inventory.inventory_value) AS inventory_value,
       avg(sales.revenue / NULLIF(inventory.inventory_value, 0)) AS mean_sales_to_inventory
FROM sales JOIN inventory
  ON inventory.supplier_key = sales.supplier_key AND inventory.brand = sales.brand
GROUP BY sales.nation, sales.brand
HAVING count(*) >= 2;
