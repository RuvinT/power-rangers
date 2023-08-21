import pandas as pd
import datetime
import pymysql


def insert_data(start_datetime, end_datetime):
    df_demand = pd.read_csv("Data/Demand_2023.csv", skiprows=3)
    df_supply = pd.read_csv("Data/Supply_2023.csv")
    df_price = pd.read_csv("Data/HOEP_2023.csv", skiprows=3)

    while start_datetime <= end_datetime:
        current_datetime = start_datetime
        current_date = current_datetime.date()
        current_hour = current_datetime.hour + 1
        # extract data
        market_demand = list(df_demand[(pd.to_datetime(df_demand['Date']) == pd.to_datetime(current_date)) & (
                df_demand['Hour'] == current_hour)][
                                 'Market Demand'])[0]
        ontario_demand = list(df_demand[(pd.to_datetime(df_demand['Date']) == pd.to_datetime(current_date)) & (
                df_demand['Hour'] == current_hour)][
                                  'Ontario Demand'])[0]
        nuclear_supply = list(df_supply[(pd.to_datetime(df_supply['Date']) == pd.to_datetime(current_date)) & (
                df_supply['Hour'] == current_hour)]['NUCLEAR'])[0]
        gas_supply = list(df_supply[(pd.to_datetime(df_supply['Date']) == pd.to_datetime(current_date)) & (
                df_supply['Hour'] == current_hour)]['GAS'])[0]
        hydro_supply = list(df_supply[(pd.to_datetime(df_supply['Date']) == pd.to_datetime(current_date)) & (
                df_supply['Hour'] == current_hour)]['HYDRO'])[0]
        wind_supply = list(df_supply[(pd.to_datetime(df_supply['Date']) == pd.to_datetime(current_date)) & (
                df_supply['Hour'] == current_hour)]['WIND'])[0]
        solar_supply = list(df_supply[(pd.to_datetime(df_supply['Date']) == pd.to_datetime(current_date)) & (
                df_supply['Hour'] == current_hour)]['SOLAR'])[0]
        biofuel_supply = list(df_supply[(pd.to_datetime(df_supply['Date']) == pd.to_datetime(current_date)) & (
                df_supply['Hour'] == current_hour)]['BIOFUEL'])[0]
        total_supply = list(df_supply[(pd.to_datetime(df_supply['Date']) == pd.to_datetime(current_date)) & (
                df_supply['Hour'] == current_hour)]['Total Output'])[0]
        hoep = list(df_price[(pd.to_datetime(df_supply['Date']) == pd.to_datetime(current_date)) & (
                df_price['Hour'] == current_hour)]['HOEP'])[0]

        # create dictionary from extracted values
        data = dict()
        data['Datetime'] = datetime.datetime(current_date.year, current_date.month, current_date.day, current_hour - 1,
                                             0, 0)
        data['MarketDemand(MW)'] = int(market_demand)
        data['OntarioDemand(MW)'] = int(ontario_demand)
        data['NuclearSupply(MW)'] = int(nuclear_supply)
        data['GasSupply(MW)'] = int(gas_supply)
        data['HydroSupply(MW)'] = int(hydro_supply)
        data['WindSupply(MW)'] = int(wind_supply)
        data['SolarSupply(MW)'] = int(solar_supply)
        data['BiofuelSupply(MW)'] = int(biofuel_supply)
        data['TotalSupply(MW)'] = int(total_supply)
        data['HOEP($/MWh)'] = float(hoep)

        # insert data using api
        db = pymysql.connect(host="", user="admin",
                             password="", database="mydatabase")
        cursor = db.cursor()
        query = """
        INSERT INTO `energy` (`Date`, `Hour`, `Market_Demand`, `Ontario_Demand`, `Nuclear`, `Gas`, `Hydro`, `Wind`, `Solar`, `Biofuel`, `Total_Output`, `HOEP`, `Hour_1_Predispatch`, `Hour_2_Predispatch`, `Hour_3_Predispatch`, `OR_10_Min_Sync`, `OR_10_Min_non-sync`, `OR_30_Min`)
        VALUES (%(date)s, %(hour)s, %(market_demand)s, %(ontario_demand)s, %(nuclear)s, %(gas)s, %(hydro)s, %(wind)s, %(solar)s, %(biofuel)s, %(total_output)s, %(hoep)s, NULL, NULL, NULL, NULL, NULL, NULL)
        """

        cursor.execute(query, {
            'date': data['Datetime'].date(),
            'hour': data['Datetime'].hour,
            'market_demand': data['MarketDemand(MW)'],
            'ontario_demand': data['OntarioDemand(MW)'],
            'nuclear': data['NuclearSupply(MW)'],
            'gas': data['GasSupply(MW)'],
            'hydro': data['HydroSupply(MW)'],
            'wind': data['WindSupply(MW)'],
            'solar': data['SolarSupply(MW)'],
            'biofuel': data['BiofuelSupply(MW)'],
            'total_output': data['TotalSupply(MW)'],
            'hoep': data['HOEP($/MWh)']
        })
        db.commit()

        start_datetime += datetime.timedelta(hours=1)


# convert this to insert data from any datetime to any datetime
insert_data(datetime.datetime(2023, 5, 21, 0, 0, 0), datetime.datetime(2023, 5, 22, 20, 0, 0))
