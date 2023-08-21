CREATE TABLE yearly_data AS
SELECT SUBSTR(month, 1, 4) AS year,
SUM("MarketDemand(MW)") AS "MarketDemand(MW)",
SUM("OntarioDemand(MW)") AS "OntarioDemand(MW)",
SUM("NuclearSupply(MW)") AS "NuclearSupply(MW)",
SUM("GasSupply(MW)") AS "GasSupply(MW)",
SUM("HydroSupply(MW)") AS "HydroSupply(MW)",
SUM("WindSupply(MW)") AS "WindSupply(MW)",
SUM("SolarSupply(MW)") AS "SolarSupply(MW)",
SUM("BiofuelSupply(MW)") AS "BiofuelSupply(MW)",
SUM("TotalSupply(MW)") AS "TotalSupply(MW)",
AVG("HOEP($/MWh)") AS "HOEP($/MWh)"
FROM monthly_data
GROUP BY year