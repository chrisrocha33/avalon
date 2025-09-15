SELECT TABLE_SCHEMA, TABLE_NAME
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_TYPE = 'BASE TABLE';

SELECT c.oid,
       n.nspname   AS schema_name,
       c.relname   AS table_name,
       c.relkind,
       pg_catalog.pg_get_userbyid(c.relowner) AS owner
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind = 'r'  -- only ordinary tables
  AND n.nspname NOT IN ('pg_catalog', 'information_schema')
ORDER BY c.oid DESC;  -- higher OID ≈ newer

SELECT series_id, title, last_updated, observation_start, category_names
FROM fred_series_catalog
WHERE frequency_short = 'M'
  AND category_names = '["Consumer Price Indexes (CPI and PCE)"]'
  AND title NOT ILIKE '%DISCONTINUED%'
  AND title ILIKE '%RESEARCH%'





SELECT 
    (SELECT COUNT(*) FROM fred_series_catalog) *
    (SELECT COUNT(*) 
     FROM information_schema.columns 
     WHERE table_schema = 'public' 
       AND table_name = 'fred_series_catalog') 
AS total_cells;

SELECT * FROM fred_series_catalog
WHERE series_id = 'TLMFGCONS'

WITH y AS (
  SELECT CAST("Date" AS date) AS dt, "DGS10", "DGS30"
  FROM "us_Yields_Rates"
),
pi AS (
  SELECT CAST("Date" AS date) AS dt, "CPIAUCSL"
  FROM "us_Prices_Inflation"
),
lw AS (
  SELECT CAST("Date" AS date) AS dt, "PAYEMS", "UNRATE"
  FROM "us_Labor_Wages"
),
io AS (
  SELECT CAST("Date" AS date) AS dt, "INDPRO"
  FROM "us_Production_Output"
),
gm AS (
  SELECT
    CAST("Date" AS date) AS dt,
    "^GSPC","EURCAD=X","^MXX","MXN=X","^BVSP","^MERV","^IPSA","^STOXX50E","^N100","^XDE",
    "EURUSD=X","^FTSE","^BUK100P","GBPUSD=X","^GDAXI","^FCHI","^BFX","EURCHF=X","EURSEK=X",
    "EURHUF=X","MOEX.ME","RUB=X","000001.SS","CNY=X","^N225","JPY=X","EURJPY=X","^HSI","HKD=X",
    "^STI","SGD=X","^BSESN","INR=X","^JKSE","IDR=X","^KLSE","MYR=X","^KS11","KRW=X","^TWII",
    "PHP=X","THB=X","^AXJO","^AORD","AUDUSD=X","^NZ50","NZDUSD=X","^TA125.TA","^JN0U.JO",
    "ZAR=X","DX-Y.NYB","^125904-USD-STRD","GBPJPY=X"
  FROM "global_markets_df"
)
SELECT
  y.dt AS "Date",
  y."DGS10", y."DGS30",
  pi."CPIAUCSL",
  lw."PAYEMS", lw."UNRATE",
  io."INDPRO",
  gm."^GSPC", gm."EURCAD=X", gm."^MXX", gm."MXN=X", gm."^BVSP", gm."^MERV", gm."^IPSA",
  gm."^STOXX50E", gm."^N100", gm."^XDE", gm."EURUSD=X", gm."^FTSE", gm."^BUK100P",
  gm."GBPUSD=X", gm."^GDAXI", gm."^FCHI", gm."^BFX", gm."EURCHF=X", gm."EURSEK=X",
  gm."EURHUF=X", gm."MOEX.ME", gm."RUB=X", gm."000001.SS", gm."CNY=X", gm."^N225",
  gm."JPY=X", gm."EURJPY=X", gm."^HSI", gm."HKD=X", gm."^STI", gm."SGD=X", gm."^BSESN",
  gm."INR=X", gm."^JKSE", gm."IDR=X", gm."^KLSE", gm."MYR=X", gm."^KS11", gm."KRW=X",
  gm."^TWII", gm."PHP=X", gm."THB=X", gm."^AXJO", gm."^AORD", gm."AUDUSD=X", gm."^NZ50",
  gm."NZDUSD=X", gm."^TA125.TA", gm."^JN0U.JO", gm."ZAR=X", gm."DX-Y.NYB",
  gm."^125904-USD-STRD", gm."GBPJPY=X"
FROM y
LEFT JOIN pi ON pi.dt = y.dt
LEFT JOIN lw ON lw.dt = y.dt
LEFT JOIN io ON io.dt = y.dt
LEFT JOIN gm ON gm.dt = y.dt
ORDER BY y.dt;


