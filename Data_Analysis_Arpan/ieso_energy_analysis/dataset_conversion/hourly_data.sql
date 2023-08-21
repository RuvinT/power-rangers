CREATE TABLE hourly_data AS 
SELECT datetime(Datetime) AS Datetime,
"MarketDemand(MW)",
"OntarioDemand(MW)",
CAST("NuclearSupply(MW)" AS REAL) AS "NuclearSupply(MW)",
CAST("GasSupply(MW)" AS REAL) AS "GasSupply(MW)",
CAST("HydroSupply(MW)" AS REAL) AS "HydroSupply(MW)",
CAST("WindSupply(MW)" AS REAL) AS "WindSupply(MW)",
CAST("SolarSupply(MW)" AS REAL) AS "SolarSupply(MW)",
CAST("BiofuelSupply(MW)" AS REAL) AS "BiofuelSupply(MW)",
CAST("TotalSupply(MW)" AS REAL) AS "TotalSupply(MW)",
"HOEP($/MWh)" FROM default_data