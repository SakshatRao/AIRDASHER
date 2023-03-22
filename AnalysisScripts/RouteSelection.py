import numpy as np
import pandas as pd

import geopy.distance

import shutil
import os
from collections import OrderedDict

import time

def RouteSelection_Script(selected_city, AIRPORTS, general_params, preprocessor, tier_1_2_cities, output_save_path, plotly_save_path):

    PRESENT_YEAR = general_params['PRESENT_YEAR']
    FORECAST_YEAR = general_params['FORECAST_YEAR']
    SAMPLE_NAME = general_params['SAMPLE_NAME']

    preprocessor.network_data = preprocessor.all_samples_network_data[SAMPLE_NAME]

    distance_time_curve_fit = np.polyfit(preprocessor.all_network_data['Distance'].values, preprocessor.all_network_data['Time'].values, 1)

    def get_route_passenger_traffic_data(cities):
        city_to_city_mapping = dict(zip(preprocessor.city_mapping['DomesticPassengerTraffic_City'], preprocessor.city_mapping['City']))
        route_traffic = []
        for idx, row in preprocessor.total_domestic_data.iterrows():
            if((row['FROM'] in city_to_city_mapping) and (row['TO'] in city_to_city_mapping)):
                route_traffic.append([city_to_city_mapping[row['FROM']], city_to_city_mapping[row['TO']], row['PASSENGERS']])
        route_traffic_df = pd.DataFrame(route_traffic, columns = ['From', 'To', 'Passengers_Target'])
        return route_traffic_df

    route_traffic_df = get_route_passenger_traffic_data(tier_1_2_cities)
    route_traffic_df['Year'] = pd.Series([PRESENT_YEAR] * route_traffic_df.shape[0])
    route_traffic_df['Connecting'] = pd.Series([''] * route_traffic_df.shape[0])

    def get_railways_info_features(raw_route_traffic_df):
        route_traffic_df = raw_route_traffic_df.copy()
        railway_info = []
        city_to_district_mapping = dict(zip(preprocessor.city_mapping['City'].values, preprocessor.city_mapping['StationCodeData_District'].values))
        for idx, row in route_traffic_df.iterrows():
            from_district = city_to_district_mapping[row['From']]
            to_district = city_to_district_mapping[row['To']]
            if(from_district not in preprocessor.city_to_city_train_dict):
                railway_info.append([row['From'], row['To'], row['Year']] + [0, np.nan, np.nan])
            else:
                if(to_district not in preprocessor.city_to_city_train_dict[from_district]):
                    railway_info.append([row['From'], row['To'], row['Year']] + [0, np.nan, np.nan])
                else:
                    route_railway_info = preprocessor.city_to_city_train_dict[from_district][to_district]
                    num_trains = len(route_railway_info)
                    avg_duration = np.nanmean([x['duration'] for x in route_railway_info])
                    third_ac = np.nansum([x['third_ac'] for x in route_railway_info])
                    chair_car = np.nansum([x['chair_car'] for x in route_railway_info])
                    first_class = np.nansum([x['first_class'] for x in route_railway_info])
                    sleeper = np.nansum([x['sleeper'] for x in route_railway_info])
                    second_ac = np.nansum([x['second_ac'] for x in route_railway_info])
                    first_ac = np.nansum([x['first_ac'] for x in route_railway_info])
                    # Based on https://en.wikipedia.org/wiki/Indian_Railways_coaching_stock & https://www.quora.com/What-is-the-capacity-of-normal-Indian-passenger-train
                    #   first_ac -> 20 (1x20)
                    #   second_ac -> 100 (2x50)
                    #   third_ax -> 180 (3x60)
                    #   chair_car -> 150 (75x2)
                    #   sleeper -> 840 (12x70)
                    #   first_class -> 20 (1x20)
                    capacity = (
                        first_ac * 20 +
                        second_ac * 100 +
                        third_ac * 180 +
                        chair_car * 150 +
                        sleeper * 840 +
                        first_class * 20
                    )
                    # Forecasting growth of railways
                    #    -> Based on https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiQp_S3qNP9AhXIwTgGHYuaB5EQFnoECA8QAQ&url=https%3A%2F%2Findianrailways.gov.in%2Frailwayboard%2Fuploads%2Fdirectorate%2Fstat_econ%2FAnnual-Reports-2020-2021%2FAnnual-Report-English.pdf&usg=AOvVaw2YMsSpEfqTOjBD13N-ZsJl,
                    #    -> On average, over the past decade, number of passenger coaches increased by ~1000 per year while total running track increased by ~1000 km per year
                    #    -> On average, growth rate in railway connectivity & passenger capacity was ~2% per year
                    num_trains = num_trains + num_trains * (2 / 100) * (row['Year'] - PRESENT_YEAR)
                    capacity = capacity + capacity * (2 / 100) * (row['Year'] - PRESENT_YEAR)
                    railway_info.append([row['From'], row['To'], row['Year']] + [num_trains, avg_duration, capacity])
        railway_info_df = pd.DataFrame(railway_info, columns = ['From', 'To', 'Year', 'NumTrains_Railways', 'Duration_Railways', 'Capacity_Railways'])
        route_traffic_df = pd.merge(route_traffic_df, railway_info_df, on = ['From', 'To', 'Year'], how = 'left')
        return route_traffic_df

    route_traffic_df = get_railways_info_features(route_traffic_df)

    OVERLAY_TIME = 120
    airport_to_coords_mapping = dict(zip(preprocessor.city_mapping['AirRouteData_AirportCode'].values, preprocessor.city_mapping['Airport_City_Coords'].values))
    def get_timing(from_airport, to_airport):
            route_timing_data = preprocessor.all_network_data[(preprocessor.all_network_data['From'] == from_airport) & (preprocessor.all_network_data['To'] == to_airport)]
            if(route_timing_data.shape[0] == 0):
                from_airport_coords = airport_to_coords_mapping[from_airport]
                from_airport_coords_lat = float(from_airport_coords.split(',')[0])
                from_airport_coords_lon = float(from_airport_coords.split(',')[1])
                from_airport_coords = (from_airport_coords_lat, from_airport_coords_lon)
                to_airport_coords = airport_to_coords_mapping[to_airport]
                to_airport_coords_lat = float(to_airport_coords.split(',')[0])
                to_airport_coords_lon = float(to_airport_coords.split(',')[1])
                to_airport_coords = (to_airport_coords_lat, to_airport_coords_lon)
                route_distance = geopy.distance.geodesic(from_airport_coords, to_airport_coords).miles
                route_timing = route_distance * distance_time_curve_fit[0] + distance_time_curve_fit[1]
            else:
                assert(route_timing_data['Time'].nunique() == 1)
                route_distance = route_timing_data['Distance'].mean()
                route_timing = route_timing_data['Time'].mean()
            return route_timing, route_distance

    def get_route_timing_features(raw_route_traffic_df):
        
        route_traffic_df = raw_route_traffic_df.copy()
        route_timings = []
        city_to_airport_mapping = dict(zip(preprocessor.city_mapping['City'].values, preprocessor.city_mapping['AirRouteData_AirportCode'].values))
        for idx, row in route_traffic_df.iterrows():
            from_airport = city_to_airport_mapping[row['From']]
            to_airport = city_to_airport_mapping[row['To']]
            connecting = row['Connecting']
            if(connecting == ''):
                route_timing = get_timing(from_airport, to_airport)[0]
            else:
                connecting_airport = city_to_airport_mapping[connecting]
                route_timing = get_timing(from_airport, connecting_airport)[0] + get_timing(connecting_airport, to_airport)[0] + OVERLAY_TIME
            route_timings.append(route_timing)
        route_traffic_df['Duration_AirRoute'] = pd.Series(route_timings)
        route_traffic_df['NumTrains_Railways'] = route_traffic_df['NumTrains_Railways'].fillna(0)
        return route_traffic_df

    route_traffic_df = get_route_timing_features(route_traffic_df)

    pca_vals_df = preprocessor.present_features.copy()
    pca_cols = ['City_' + x for x in pca_vals_df.columns if x != 'City']
    pca_vals_df.columns = pca_cols + ['City']
    pca_vals_df['Year'] = pd.Series([PRESENT_YEAR] * pca_vals_df.shape[0])

    pca_vals_forecasted_df = {}
    for year in range(PRESENT_YEAR + 1, FORECAST_YEAR + 1):
        pca_vals_forecasted_year_df = pd.read_csv(f"{output_save_path}/Forecasted_Features/{year}.csv")
        pca_vals_forecasted_year_df.columns = ['City_' + x if x != 'City' else x for x in pca_vals_forecasted_year_df.columns]
        pca_vals_forecasted_year_df['Year'] = pd.Series([year] * pca_vals_forecasted_year_df.shape[0])
        pca_vals_forecasted_df[year] = pca_vals_forecasted_year_df

    total_pca_vals_df = pca_vals_df
    for year in pca_vals_forecasted_df:
        total_pca_vals_df = pd.concat([total_pca_vals_df, pca_vals_forecasted_df[year]], axis = 0)

    def get_pca_vals_features(raw_route_traffic_df):
        route_traffic_df = raw_route_traffic_df.copy()
        route_traffic_df = pd.merge(route_traffic_df, total_pca_vals_df, left_on = ['From', 'Year'], right_on = ['City', 'Year'], how = 'left')
        route_traffic_df = pd.merge(route_traffic_df, total_pca_vals_df, left_on = ['To', 'Year'], right_on = ['City', 'Year'], how = 'left', suffixes = ('_pca_vals_FROM', '_pca_vals_TO'))
        
        route_traffic_df = route_traffic_df.drop(['City_pca_vals_TO', 'City_pca_vals_FROM'], axis = 1)
        for col in pca_cols:
            route_traffic_df[f'{col}_avg_PCA'] = (route_traffic_df[f'{col}_pca_vals_FROM'] + route_traffic_df[f'{col}_pca_vals_TO']) / 2
            route_traffic_df[f'{col}_diff_PCA'] = (route_traffic_df[f'{col}_pca_vals_FROM'] - route_traffic_df[f'{col}_pca_vals_TO'])
            route_traffic_df = route_traffic_df.drop([f'{col}_pca_vals_FROM', f'{col}_pca_vals_TO', f'{col}_pca_vals_FROM', f'{col}_pca_vals_TO'], axis = 1)
        
        return route_traffic_df

    route_traffic_df = get_pca_vals_features(route_traffic_df)

    y_features = [x for x in route_traffic_df.columns if x.endswith('_Target')]
    X_features = [x for x in route_traffic_df.columns if (x.endswith('_PCA')) or (x.endswith('_AirRoute')) or (x.endswith('_Railways'))]

    # Trimming data for model training - Removing routes having no target variable
    # Standardizing data
    valid_route_traffic_df = route_traffic_df.copy()
    cols_standardization_vals = {}
    for col_idx, col in enumerate(valid_route_traffic_df.columns):
        if((col in X_features) or (col in y_features)):
            if(col in y_features):
                to_drop_idx = pd.isnull(valid_route_traffic_df[col])
                to_drop_idx = to_drop_idx[to_drop_idx == True].index
                valid_route_traffic_df = valid_route_traffic_df.drop(to_drop_idx, axis = 0)
            elif(col in X_features):
                col_mean = np.nanmean(valid_route_traffic_df[col].values)
                valid_route_traffic_df[col] = valid_route_traffic_df[col].fillna(col_mean)
            col_mean = valid_route_traffic_df[col].mean()
            col_std = valid_route_traffic_df[col].std()
            valid_route_traffic_df[col] = (valid_route_traffic_df[col] - col_mean) / (col_std + 1e-20)
            cols_standardization_vals[col] = {'mean': col_mean, 'std': col_std}

    class LinearModel:
        def __init__(self, coefs):
            self.coefs = coefs
        def predict(self, X):
            assert(len(self.coefs) == X.shape[1] + 1)
            return np.dot([*self.coefs.values()][1:], X.transpose()) + self.coefs['intercept']

    model = LinearModel(preprocessor.RouteSelection_model_coefs)

    uniq_hubs = preprocessor.network_data[preprocessor.network_data['FromHub'] == 1]['From'].unique()

    def get_route_demand_forecasts(SELECTED_CITY, SELECTED_HUB_AIRPORT):
        selected_hub_city = preprocessor.city_mapping[preprocessor.city_mapping['AirRouteData_AirportCode'] == SELECTED_HUB_AIRPORT].iloc[0]['City']
        connecting_airports = []
        for airport in AIRPORTS[SELECTED_HUB_AIRPORT].to_airport_list:
            if(airport not in connecting_airports):
                connecting_airports.append(airport)
        for airport in AIRPORTS[SELECTED_HUB_AIRPORT].from_airport_list:
            if(airport not in connecting_airports):
                connecting_airports.append(airport)
        
        expected_route_traffic_df = []
        for year in np.arange(PRESENT_YEAR, FORECAST_YEAR + 1):
            expected_route_traffic_df.append([SELECTED_CITY, selected_hub_city, year, ''])
            airport_to_city_mapping = dict(zip(preprocessor.city_mapping['AirRouteData_AirportCode'].values, preprocessor.city_mapping['City'].values))
            for airport in connecting_airports:
                expected_route_traffic_df.append([SELECTED_CITY, airport_to_city_mapping[airport.airport_info['Name']], year, selected_hub_city])
            expected_route_traffic_df.append([selected_hub_city, SELECTED_CITY, year, ''])
            airport_to_city_mapping = dict(zip(preprocessor.city_mapping['AirRouteData_AirportCode'].values, preprocessor.city_mapping['City'].values))
            for airport in connecting_airports:
                expected_route_traffic_df.append([airport_to_city_mapping[airport.airport_info['Name']], SELECTED_CITY, year, selected_hub_city])
        expected_route_traffic_df = pd.DataFrame(expected_route_traffic_df, columns = ['From', 'To', 'Year', 'Connecting'])
        
        expected_route_traffic_df = get_railways_info_features(expected_route_traffic_df)
        expected_route_traffic_df = get_route_timing_features(expected_route_traffic_df)
        expected_route_traffic_df = get_pca_vals_features(expected_route_traffic_df)
        duration_in = expected_route_traffic_df[(expected_route_traffic_df['From'] == selected_hub_city) & (expected_route_traffic_df['To'] == SELECTED_CITY)].iloc[0]['Duration_AirRoute']
        duration_out = expected_route_traffic_df[(expected_route_traffic_df['To'] == selected_hub_city) & (expected_route_traffic_df['From'] == SELECTED_CITY)].iloc[0]['Duration_AirRoute']
        
        for col_idx, col in enumerate(expected_route_traffic_df.columns):
            if(col in X_features):
                col_mean = cols_standardization_vals[col]['mean']
                col_std = cols_standardization_vals[col]['std']
                expected_route_traffic_df[col] = expected_route_traffic_df[col].fillna(col_mean)
                expected_route_traffic_df[col] = (expected_route_traffic_df[col] - col_mean) / (col_std + 1e-20)
        
        data_X = expected_route_traffic_df[X_features]
        target_mean = cols_standardization_vals[y_features[0]]['mean']
        target_std = cols_standardization_vals[y_features[0]]['std']
        pred = model.predict(data_X) * target_std + (target_mean)
        expected_route_traffic_df['ForecastedDemand'] = pd.Series(pred)
        
        # Domestic Passenger Traffic would include local + connecting demand, hence excluding adding connecting demand
        CONNECTING_DEMAND_FACTOR = 0
        def adjust_connecting_demand(row):
            if(row['Connecting'] == ''):
                return row['ForecastedDemand']
            else:
                return row['ForecastedDemand'] * CONNECTING_DEMAND_FACTOR
        expected_route_traffic_df['AdjustedForecastedDemand'] = expected_route_traffic_df.apply(adjust_connecting_demand, axis = 1)
        
        # Domestic Passenger Traffic would include local + connecting demand, hence excluding adding connecting demand
        CONNECTING_DEMAND_FACTOR = 0
        def adjust_connecting_demand(row):
            if(row['Connecting'] == ''):
                return row['ForecastedDemand']
            else:
                return row['ForecastedDemand'] * CONNECTING_DEMAND_FACTOR
        expected_route_traffic_df['AdjustedForecastedDemand'] = expected_route_traffic_df.apply(adjust_connecting_demand, axis = 1)
        
        in_total_traffic = expected_route_traffic_df[expected_route_traffic_df['To'] == SELECTED_CITY]
        in_total_traffic = in_total_traffic.groupby('Year')['AdjustedForecastedDemand'].aggregate('sum').reset_index(drop = False)
        out_total_traffic = expected_route_traffic_df[expected_route_traffic_df['From'] == SELECTED_CITY]
        out_total_traffic = out_total_traffic.groupby('Year')['AdjustedForecastedDemand'].aggregate('sum').reset_index(drop = False)
        in_out_total_traffic = pd.merge(in_total_traffic, out_total_traffic, on = 'Year', suffixes = ('_InTraffic', '_OutTraffic'))
        
        actual_present_out_traffic = valid_route_traffic_df[(valid_route_traffic_df['From'] == SELECTED_CITY) & (valid_route_traffic_df['To'] == selected_hub_city)]
        if(actual_present_out_traffic.shape[0] > 0):
            assert(actual_present_out_traffic.shape[0] == 1)
            actual_present_out_traffic = actual_present_out_traffic.iloc[0]['Passengers_Target'] * target_std + target_mean
            out_traffic_adjustment = actual_present_out_traffic - in_out_total_traffic.iloc[0]['AdjustedForecastedDemand_OutTraffic']
            in_out_total_traffic['AdjustedForecastedDemand_OutTraffic'] = in_out_total_traffic['AdjustedForecastedDemand_OutTraffic'] + out_traffic_adjustment
        actual_present_in_traffic = valid_route_traffic_df[(valid_route_traffic_df['To'] == SELECTED_CITY) & (valid_route_traffic_df['From'] == selected_hub_city)]
        if(actual_present_in_traffic.shape[0] > 0):
            actual_present_in_traffic = actual_present_in_traffic.iloc[0]['Passengers_Target'] * target_std + target_mean
            in_traffic_adjustment = actual_present_in_traffic - in_out_total_traffic.iloc[0]['AdjustedForecastedDemand_InTraffic']
            in_out_total_traffic['AdjustedForecastedDemand_InTraffic'] = in_out_total_traffic['AdjustedForecastedDemand_InTraffic'] + in_traffic_adjustment
        
        in_out_total_traffic.to_csv(f"{output_save_path}/Forecasted_Route_Demand/City{SELECTED_CITY}_Hub{SELECTED_HUB_AIRPORT}.csv", index = None)
        return in_out_total_traffic, (duration_in, duration_out)

    airport_to_city_mapping = dict(zip(preprocessor.city_mapping['AirRouteData_AirportCode'].values, preprocessor.city_mapping['City'].values))
    city_to_airport_mapping = dict(zip(preprocessor.city_mapping['City'].values, preprocessor.city_mapping['AirRouteData_AirportCode'].values))
    all_airports_in_network = [airport_to_city_mapping[AIRPORTS[x].airport_info['Name']] for x in AIRPORTS]

    shutil.rmtree(f"{output_save_path}/Forecasted_Route_Demand/", ignore_errors = True)
    os.mkdir(f"{output_save_path}/Forecasted_Route_Demand/")
    route_info_df = []
    city = selected_city
    for hub in uniq_hubs:
        SELECTED_CITY = city
        SELECTED_HUB_AIRPORT = hub
        route_forecasted_demands, durations = get_route_demand_forecasts(SELECTED_CITY, SELECTED_HUB_AIRPORT)
        assert(route_forecasted_demands.iloc[0]['Year'] == PRESENT_YEAR)
        assert(route_forecasted_demands.iloc[route_forecasted_demands.shape[0] - 1]['Year'] == FORECAST_YEAR)

        selected_city_airport = city_to_airport_mapping[SELECTED_CITY]
        route_network_data_out = preprocessor.all_network_data[(preprocessor.all_network_data['From'] == selected_city_airport) & (preprocessor.all_network_data['To'] == SELECTED_HUB_AIRPORT)]
        route_network_data_in = preprocessor.all_network_data[(preprocessor.all_network_data['To'] == selected_city_airport) & (preprocessor.all_network_data['From'] == SELECTED_HUB_AIRPORT)]
        if(route_network_data_out.shape[0] == 0):
            NUM_OUT_MARKET = 0
            PRICE_OUT_MARKET = "N/A"
        else:
            PRICE_OUT_MARKET = f"${int(route_network_data_out['Cheapest Price'].mean())}"
            NUM_OUT_MARKET = int(route_network_data_out['Number of Flights'].sum())
        if(route_network_data_in.shape[0] == 0):
            NUM_IN_MARKET = 0
            PRICE_IN_MARKET = "N/A"
        else:
            PRICE_IN_MARKET = f"${int(route_network_data_in['Cheapest Price'].mean())}"
            NUM_IN_MARKET = int(route_network_data_in['Number of Flights'].sum())
        
        DISTANCE = (get_timing(selected_city_airport, SELECTED_HUB_AIRPORT)[1] + get_timing(SELECTED_HUB_AIRPORT, selected_city_airport)[1]) / 2.0

        route_info_df.append([
            SELECTED_CITY, SELECTED_HUB_AIRPORT,
            durations[0], durations[1],
            route_forecasted_demands.iloc[0]['AdjustedForecastedDemand_InTraffic'],
            route_forecasted_demands.iloc[0]['AdjustedForecastedDemand_OutTraffic'],
            route_forecasted_demands.iloc[route_forecasted_demands.shape[0] - 1]['AdjustedForecastedDemand_InTraffic'],
            route_forecasted_demands.iloc[route_forecasted_demands.shape[0] - 1]['AdjustedForecastedDemand_OutTraffic'],
            NUM_OUT_MARKET, NUM_IN_MARKET,
            PRICE_OUT_MARKET, PRICE_IN_MARKET,
            DISTANCE
        ])
    route_info_df = pd.DataFrame(route_info_df, columns = [
        'City', 'Hub',
        'IncomingFlightDuration', 'OutgoingFlightDuration',
        'PresentYearInForecast', 'PresentYearOutForecast',
        'ForecastYearInForecast', 'ForecastYearOutForecast',
        'NUMBER_PLANES_OUT_MARKET', 'NUMBER_PLANES_IN_MARKET',
        'PRICE_OUT_MARKET', 'PRICE_IN_MARKET',
        'DISTANCE'
    ])
    route_info_df['GrowthIn'] = (route_info_df['ForecastYearInForecast'] - route_info_df['PresentYearInForecast']) / (route_info_df['PresentYearInForecast'] + 1e-12) * 100
    route_info_df['GrowthOut'] = (route_info_df['ForecastYearOutForecast'] - route_info_df['PresentYearOutForecast']) / (route_info_df['PresentYearOutForecast'] + 1e-12) * 100
    route_info_df['AvgGrowth'] = (route_info_df['GrowthIn'] + route_info_df['GrowthOut']) / 2.0
    route_info_df = route_info_df.sort_values('AvgGrowth', ascending = False)
    route_info_df['Route'] = route_info_df.apply(lambda x: x['City'] + '-' + x['Hub'], axis = 1)
    route_info_df = OrderedDict(route_info_df.set_index('Route').head(5).to_dict(orient = 'index'))
    
    return route_info_df

# # Testing
# from CitySelection import CitySelection_Script
# tier_1_2_cities = [
#     'Ahmedabad', 'Bengaluru', 'Mumbai', 'Pune', 'Chennai', 'Hyderabad', 'Kolkata', 'Delhi', 'Visakhapatnam', 'Guwahati', 'Patna',
#     'Raipur', 'Gurugram', 'Shimla', 'Jamshedpur', 'Thiruvananthapuram', 'Bhopal', 'Bhubaneswar', 'Amritsar', 'Jaipur', 'Lucknow', 'Dehradun'
# ]
# t1 = time.time()
# from PreProcessor import PreProcessor
# preprocessor = PreProcessor(tier_1_2_cities, "./PreProcessed_Datasets")
# print(f"Time taken for preprocessor: {(time.time() - t1):.2f}s")
# general_params = {
#     'PRESENT_YEAR': 2023,
#     'FORECAST_YEAR': 2033,
#     'SAMPLE_NAME': 'Sample1'
# }
# t1 = time.time()
# most_growth_cities, airports = CitySelection_Script(general_params, preprocessor, tier_1_2_cities, './Final_Analysis_Scripts/Temporary_Outputs', './Final_Analysis_Scripts/Temporary_Outputs')
# print(f"Time taken for CitySelection: {(time.time() - t1):.2f}s")
# print(airports)
# print([airports[x].airport_info for x in airports])
# t1 = time.time()
# print(most_growth_cities)
# selected_city = [x for x in most_growth_cities][0]
# most_growth_routes = RouteSelection_Script(selected_city, airports, general_params, preprocessor, tier_1_2_cities, './Final_Analysis_Scripts/Temporary_Outputs', './Final_Analysis_Scripts/Temporary_Outputs')
# print(f"Time taken for RouteSelection: {(time.time() - t1):.2f}s")
# print(most_growth_routes)