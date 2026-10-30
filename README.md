# BI_DashBoard_Visual
Dashboard Visualization with KPIS of a fictional office retail store

What the dashboard shows
The dashboard moves from the big picture down to the details:
1. Headline KPIs: total sales, number of orders, average order value (AOV), average shipping time, and the compound annual growth rate (CAGR) of sales.
2. Sales and order trends: monthly development with trend lines, to separate the long-term direction from seasonal swings.
3. Product view: AOV by category and quarterly sales by year.
4. Geography and operations: sales concentration by state (top 10 versus the rest) and shipping reliability per ship mode.
5. Customer segments: total sales per segment and year, ranked, with year-over-year growth.

How the KPIs are defined
A few choices ensure the numbers are reliable and comparable:
- Orders are counted by unique order ID, because each row in the data is a product line rather than an order. Counting rows would inflate the number of orders and understate the AOV.
- AOV is total sales divided by the number of orders.
- Shipping time is the number of days between order and shipment, measured once per order, so large orders do not carry more weight.
- Shipping reliability uses the 90th percentile: the number of days within which 90% of orders arrive. This reflects the customer experience better than an average.
- Growth is shown as year-over-year change and as CAGR, based on complete calendar years so the comparison is fair.

How the data is prepared
Loadind of the data, standardization the date formats, definition of the additional features (shipping time, year, quarter, and month), calculation of the the KPIs, and assembly of the visuals into a single dashboard.
