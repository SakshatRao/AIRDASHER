import numpy as np
import pandas as pd

import geopy.distance

import shutil
import os
from collections import OrderedDict

import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

import time

##################################################
# RouteSelection_Script
##################################################
# -> Script for selecting route with highest
#    growth potential
##################################################
def RouteSelection_Script(selected_city, AIRPORTS, general_params, preprocessor, tier_1_2_cities, output_save_path, plotly_save_path):

    # Fetching parameters
    PRESENT_YEAR = general_params['PRESENT_YEAR']
    FORECAST_YEAR = general_params['FORECAST_YEAR']
    SAMPLE_NAME = general_params['SAMPLE_NAME']
    ONLY_HUBS = general_params['ONLY_HUBS']

    # Fetching sample network
    preprocessor.network_data = preprocessor.all_samples_network_data[SAMPLE_NAME]

    # cities -> cities for which to collect domestic passenger data
    def get_route_passenger_traffic_data(cities):
        city_to_city_mapping = dict(zip(preprocessor.city_mapping['DomesticPassengerTraffic_City'], preprocessor.city_mapping['City']))
        route_traffic = []
        for idx, row in preprocessor.total_domestic_data.iterrows():
            if((row['FROM'] in city_to_city_mapping) and (row['TO'] in city_to_city_mapping)):    # If From or To city is a tier-I/II city, then include in data (since we have only collected PCA values for those cities)
                route_traffic.append([city_to_city_mapping[row['FROM']], city_to_city_mapping[row['TO']], row['PASSENGERS']])
        route_traffic_df = pd.DataFrame(route_traffic, columns = ['From', 'To', 'Passengers_Target'])
        return route_traffic_df

    route_traffic_df = get_route_passenger_traffic_data(tier_1_2_cities)
    route_traffic_df['Year'] = pd.Series([PRESENT_YEAR] * route_traffic_df.shape[0])
    route_traffic_df['Connecting'] = pd.Series([''] * route_traffic_df.shape[0])    # Connecting refers to whether given route has a connecting flight. Since given domestic passenger info is for direct flights, we leave this field out

    # raw_route_traffic_df -> route info for which to collect railway connectivity data
    def get_railways_info_features(raw_route_traffic_df):
        route_traffic_df = raw_route_traffic_df.copy()
        railway_info = []
        city_to_district_mapping = dict(zip(preprocessor.city_mapping['City'].values, preprocessor.city_mapping['StationCodeData_District'].values))
        for idx, row in route_traffic_df.iterrows():
            from_district = city_to_district_mapping[row['From']]
            to_district = city_to_district_mapping[row['To']]
            if(from_district not in preprocessor.city_to_city_train_dict):    # If railway data not available for FROM city
                railway_info.append([row['From'], row['To'], row['Year']] + [0, np.nan, np.nan])
            else:
                if(to_district not in preprocessor.city_to_city_train_dict[from_district]):    # If railway data not available for TO city
                    railway_info.append([row['From'], row['To'], row['Year']] + [0, np.nan, np.nan])
                else:    # If railway data availble for both FROM & TO cities
                    route_railway_info = preprocessor.city_to_city_train_dict[from_district][to_district]
                    num_trains = len(route_railway_info)    # Number of trains in given route
                    avg_duration = np.nanmean([x['duration'] for x in route_railway_info])    # Avg duration of train journey
                    # Info about availability of different coaches in train
                    third_ac = np.nansum([x['third_ac'] for x in route_railway_info])
                    chair_car = np.nansum([x['chair_car'] for x in route_railway_info])
                    first_class = np.nansum([x['first_class'] for x in route_railway_info])
                    sleeper = np.nansum([x['sleeper'] for x in route_railway_info])
                    second_ac = np.nansum([x['second_ac'] for x in route_railway_info])
                    first_ac = np.nansum([x['first_ac'] for x in route_railway_info])

                    # Based on https://en.wikipedia.org/wiki/Indian_Railways_coaching_stock & https://www.quora.com/What-is-the-capacity-of-normal-Indian-passenger-train, we can estimate total capacity of train based on availability of coaches
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
                    #    -> Based on https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiQp_S3qNP9AhXIwTgGHYuaB5EQFnoECA8QAQ&url=https%3A%2F%2Findianrailways.gov.in%2Frailwayboard%2Fuploads%2Fdirectorate%2Fstat_econ%2FAnnual-Reports-2020-2021%2FAnnual-Report-English.pdf&usg=AOvVaw2YMsSpEfqTOjBD13N-ZsJl, we can also estimate how railway features are expected to grow with time
                    #    -> On average, over the past decade, number of passenger coaches increased by ~1000 per year while total running track increased by ~1000 km per year
                    #    -> On average, growth rate in railway connectivity & passenger capacity can be considered ~2% per year
                    #    -> This linear growth rate can be used for number & capacity of trains while we assume duration of the train journey will remain constant
                    num_trains = num_trains + num_trains * (2 / 100) * (row['Year'] - PRESENT_YEAR)
                    capacity = capacity + capacity * (2 / 100) * (row['Year'] - PRESENT_YEAR)
                    railway_info.append([row['From'], row['To'], row['Year']] + [num_trains, avg_duration, capacity])
        
        # Collecting all railway data for all routes
        railway_info_df = pd.DataFrame(railway_info, columns = ['From', 'To', 'Year', 'NumTrains_Railways', 'Duration_Railways', 'Capacity_Railways'])
        route_traffic_df = pd.merge(route_traffic_df, railway_info_df, on = ['From', 'To', 'Year'], how = 'left')
        return route_traffic_df

    route_traffic_df = get_railways_info_features(route_traffic_df)

    OVERLAY_TIME = 120    # We assume for a connecting flight that on average a 2 hour overlay is expected. This is done to favor direct flights over connecting flights
    airport_to_coords_mapping = dict(zip(preprocessor.city_mapping['AirRouteData_AirportCode'].values, preprocessor.city_mapping['Airport_City_Coords'].values))

    # We fit a line between available flight distances & flight durations
    # We can use this linear expression to find durations for new routes
    distance_time_curve_fit = np.polyfit(preprocessor.all_network_data['Distance'].values, preprocessor.all_network_data['Time'].values, 1)

    # from_airport -> city/airport from which route originates
    # to_airport -> city/airport at which route ends
    def get_timing(from_airport, to_airport):
        route_timing_data = preprocessor.all_network_data[(preprocessor.all_network_data['From'] == from_airport) & (preprocessor.all_network_data['To'] == to_airport)]    # Fetching data for given route
        if(route_timing_data.shape[0] == 0):    # If data for given route does not exist, we need to estimate duration of flight
            
            # Find coordinates of FROM & TO airports
            from_airport_coords = airport_to_coords_mapping[from_airport]
            from_airport_coords_lat = float(from_airport_coords.split(',')[0])
            from_airport_coords_lon = float(from_airport_coords.split(',')[1])
            from_airport_coords = (from_airport_coords_lat, from_airport_coords_lon)
            to_airport_coords = airport_to_coords_mapping[to_airport]
            to_airport_coords_lat = float(to_airport_coords.split(',')[0])
            to_airport_coords_lon = float(to_airport_coords.split(',')[1])
            to_airport_coords = (to_airport_coords_lat, to_airport_coords_lon)

            # Calculate distance between two airports and estimate duration using our distance-duration linear expression
            route_distance = geopy.distance.geodesic(from_airport_coords, to_airport_coords).miles
            route_timing = route_distance * distance_time_curve_fit[0] + distance_time_curve_fit[1]

        else:    # If data for given route exists, simply use the duration info
            assert(route_timing_data['Time'].nunique() == 1)
            route_distance = route_timing_data['Distance'].mean()
            route_timing = route_timing_data['Time'].mean()
        
        return route_timing, route_distance    # Return total distance & duration of flight

    # raw_route_traffic_df -> route info for which to collect railway connectivity data
    def get_route_timing_features(raw_route_traffic_df):

        route_traffic_df = raw_route_traffic_df.copy()
        route_timings = []
        city_to_airport_mapping = dict(zip(preprocessor.city_mapping['City'].values, preprocessor.city_mapping['AirRouteData_AirportCode'].values))
        for idx, row in route_traffic_df.iterrows():
            from_airport = city_to_airport_mapping[row['From']]
            to_airport = city_to_airport_mapping[row['To']]
            connecting = row['Connecting']
            if(connecting == ''):    # If no connecting flight, find direct duration of flight
                route_timing = get_timing(from_airport, to_airport)[0]
            else:    # If connecting flight, add durations of both flights + overlay time
                connecting_airport = city_to_airport_mapping[connecting]
                route_timing = get_timing(from_airport, connecting_airport)[0] + get_timing(connecting_airport, to_airport)[0] + OVERLAY_TIME
            route_timings.append(route_timing)

        # Collect data for all routes
        route_traffic_df['Duration_AirRoute'] = pd.Series(route_timings)
        route_traffic_df['NumTrains_Railways'] = route_traffic_df['NumTrains_Railways'].fillna(0)
        return route_traffic_df

    route_traffic_df = get_route_timing_features(route_traffic_df)

    # Collect current PCA values of each city
    pca_vals_df = preprocessor.present_features.copy()
    pca_cols = ['City_' + x for x in pca_vals_df.columns if x != 'City']
    pca_vals_df.columns = pca_cols + ['City']
    pca_vals_df['Year'] = pd.Series([PRESENT_YEAR] * pca_vals_df.shape[0])

    # Collect forecasted PCA values of each city
    pca_vals_forecasted_df = {}
    for year in range(PRESENT_YEAR + 1, FORECAST_YEAR + 1):
        pca_vals_forecasted_year_df = pd.read_csv(f"{output_save_path}/Forecasted_Features/{year}.csv")
        pca_vals_forecasted_year_df.columns = ['City_' + x if x != 'City' else x for x in pca_vals_forecasted_year_df.columns]
        pca_vals_forecasted_year_df['Year'] = pd.Series([year] * pca_vals_forecasted_year_df.shape[0])
        pca_vals_forecasted_df[year] = pca_vals_forecasted_year_df

    total_pca_vals_df = pca_vals_df
    for year in pca_vals_forecasted_df:
        total_pca_vals_df = pd.concat([total_pca_vals_df, pca_vals_forecasted_df[year]], axis = 0)

    # raw_route_traffic_df -> route info for which to collect railway connectivity data
    def get_pca_vals_features(raw_route_traffic_df):
        route_traffic_df = raw_route_traffic_df.copy()
        # Fetch PCA values for FROM & TO cities
        route_traffic_df = pd.merge(route_traffic_df, total_pca_vals_df, left_on = ['From', 'Year'], right_on = ['City', 'Year'], how = 'left')
        route_traffic_df = pd.merge(route_traffic_df, total_pca_vals_df, left_on = ['To', 'Year'], right_on = ['City', 'Year'], how = 'left', suffixes = ('_pca_vals_FROM', '_pca_vals_TO'))
        
        route_traffic_df = route_traffic_df.drop(['City_pca_vals_TO', 'City_pca_vals_FROM'], axis = 1)
        for col in pca_cols:
            # For FROM & TO city, try to find average & difference between both cities' PCA values
            # Intuition is to cover both cases to see whether some pattern exists - route from Major to Minor city & route from both major/minor cities
            route_traffic_df[f'{col}_avg_PCA'] = (route_traffic_df[f'{col}_pca_vals_FROM'] + route_traffic_df[f'{col}_pca_vals_TO']) / 2
            route_traffic_df[f'{col}_diff_PCA'] = (route_traffic_df[f'{col}_pca_vals_FROM'] - route_traffic_df[f'{col}_pca_vals_TO'])
            route_traffic_df = route_traffic_df.drop([f'{col}_pca_vals_FROM', f'{col}_pca_vals_TO', f'{col}_pca_vals_FROM', f'{col}_pca_vals_TO'], axis = 1)
        
        return route_traffic_df

    route_traffic_df = get_pca_vals_features(route_traffic_df)

    # Collect X & y features
    y_features = [x for x in route_traffic_df.columns if x.endswith('_Target')]
    X_features = [x for x in route_traffic_df.columns if (x.endswith('_PCA')) or (x.endswith('_AirRoute')) or (x.endswith('_Railways'))]

    # Trimming data - Removing routes having no target variable
    # Also standardizing data
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

    # Simple Linear Regression model which takes linear coefficients & computes output from input
    # NOTE: We use this model instead of sklearn due to memory size issue when installing sklearn
    class LinearModel:
        def __init__(self, coefs):
            self.coefs = coefs
        def predict(self, X):
            assert(len(self.coefs) == X.shape[1] + 1)
            return np.dot([*self.coefs.values()][1:], X.transpose()) + self.coefs['intercept']
    model = LinearModel(preprocessor.RouteSelection_model_coefs)    # We basically have a fitted linear model ready now

    # Function to get forecasted predictions of several routes for several years
    # -> This function directly uses the linear model trained for Route Selection
    #
    # Input variables:
    #    SELECTED_CITY -> Which city is being considered for being added to network
    #    SELECTED_HUB_AIRPORT -> Which hub in given airline's network should we make connection to
    def get_route_demand_forecasts(SELECTED_CITY, SELECTED_HUB_AIRPORT):
        try:
            selected_hub_city = preprocessor.city_mapping[preprocessor.city_mapping['AirRouteData_AirportCode'] == SELECTED_HUB_AIRPORT].iloc[0]['City']
        except:
            print(SELECTED_CITY)
            print(SELECTED_HUB_AIRPORT)
            exit(1)

        # Find airports connected to SELECTED_HUB_AIRPORT
        # These will also contribute to overall route demand via connecting demand
        connecting_airports = []
        for airport in AIRPORTS[SELECTED_HUB_AIRPORT].to_airport_list:
            if(airport not in connecting_airports):
                connecting_airports.append(airport)
        for airport in AIRPORTS[SELECTED_HUB_AIRPORT].from_airport_list:
            if(airport not in connecting_airports):
                connecting_airports.append(airport)

        # Create all route datasets
        # -> This will include city to hub & hub to city routes
        # -> This will also include connecting airport to city & city to connecting airport (via hub)
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

        # Extract features for all direct & connecting routes
        expected_route_traffic_df = get_railways_info_features(expected_route_traffic_df)
        expected_route_traffic_df = get_route_timing_features(expected_route_traffic_df)
        expected_route_traffic_df = get_pca_vals_features(expected_route_traffic_df)

        # Extract duration of flights
        duration_in = expected_route_traffic_df[(expected_route_traffic_df['From'] == selected_hub_city) & (expected_route_traffic_df['To'] == SELECTED_CITY)].iloc[0]['Duration_AirRoute']
        duration_out = expected_route_traffic_df[(expected_route_traffic_df['To'] == selected_hub_city) & (expected_route_traffic_df['From'] == SELECTED_CITY)].iloc[0]['Duration_AirRoute']

        # Extract railway connectivity info for all routes
        railway_info_out = expected_route_traffic_df[(expected_route_traffic_df['From'] == SELECTED_CITY) & (expected_route_traffic_df['To'] == selected_hub_city) & (expected_route_traffic_df['Year'] == PRESENT_YEAR)].iloc[0]
        railway_info_in = expected_route_traffic_df[(expected_route_traffic_df['From'] == selected_hub_city) & (expected_route_traffic_df['To'] == SELECTED_CITY) & (expected_route_traffic_df['Year'] == PRESENT_YEAR)].iloc[0]
        railway_num_out = railway_info_out['NumTrains_Railways']
        railway_num_in = railway_info_in['NumTrains_Railways']
        railway_duration_out = railway_info_out['Duration_Railways']
        railway_duration_in = railway_info_in['Duration_Railways']
        railway_capacity_out = railway_info_out['Capacity_Railways']
        railway_capacity_in = railway_info_in['Capacity_Railways']

        # Function to return avg of two values if non-NA, else the non-NA value out of the two, else 'N/A'
        def check_if_both_nan(x, y):
            if((pd.isnull(x)) & (pd.isnull(y))):
                return "N/A"
            else:
                if(pd.isnull(x)):
                    return round(y, 0)
                elif(pd.isnull(y)):
                    return round(x, 0)
                else:
                    return round((x + y) // 2, 0)
        # Generalizing railway connectivity info for to & fro route in one value
        railway_num = check_if_both_nan(railway_num_in, railway_num_out)
        railway_duration = check_if_both_nan(railway_duration_in, railway_duration_out)
        railway_capacity = check_if_both_nan(railway_capacity_in, railway_capacity_out)
        
        # Standardizing features using previously collected standardization info (i.e. column means & std)
        for col_idx, col in enumerate(expected_route_traffic_df.columns):
            if(col in X_features):
                col_mean = cols_standardization_vals[col]['mean']
                col_std = cols_standardization_vals[col]['std']
                expected_route_traffic_df[col] = expected_route_traffic_df[col].fillna(col_mean)
                expected_route_traffic_df[col] = (expected_route_traffic_df[col] - col_mean) / (col_std + 1e-20)

        # Collecting data_X and making forecasts for route demands
        # We use the column standardization values for target variable to convert model outputs to forecasts
        data_X = expected_route_traffic_df[X_features]
        target_mean = cols_standardization_vals[y_features[0]]['mean']
        target_std = cols_standardization_vals[y_features[0]]['std']
        pred = model.predict(data_X) * target_std + (target_mean)
        expected_route_traffic_df['ForecastedDemand'] = pd.Series(pred)

        # Factoring in connecting demand into total demand
        #    For city C, hub H & connecting city CC,
        #    Connecting demand contribution will be dependant on -
        #        (1) How many people travelling from CC to C or from C to CC will choose our airline? (market share)
        #        (2) How many people travelling from CC to C or from C to CC will choose to have a connecting flight via H rather than a direct flight if available?
        #        (3) How many people travelling from CC to C or from C to CC will choose to have a connecting flight via H rather than some other connecting flight via some other hub?
        #
        # CONNECTING_FACTOR is factor by which total demand from CC to C or from C to CC will serve as total connecting demand to/from C
        # Each condition above will keep reducing CONNECTING_FACTOR from 100% to a smaller value
        # To keep analysis simple, let us assume -
        #    (1) Let's say average market share is 20%
        #    (2) Let's say 20% of people travel by connecting flights instead of direct flight
        #    (3) Let's say 25% of people travel by connecting flight via H rather than some other hub
        # Hence CONNECTING_FACTOR would be 1% (i.e. 1% of people wanting to travel C->CC or CC->C will travel via our airline's route C->H or H->C respectively)
        CONNECTING_DEMAND_FACTOR = 0.01
        def adjust_connecting_demand(row):
            if(row['Connecting'] == ''):
                return row['ForecastedDemand']
            else:
                return row['ForecastedDemand'] * CONNECTING_DEMAND_FACTOR
        expected_route_traffic_df['AdjustedForecastedDemand'] = expected_route_traffic_df.apply(adjust_connecting_demand, axis = 1)

        # Adding direct & connecting traffic for to & fro routes
        in_total_traffic = expected_route_traffic_df[expected_route_traffic_df['To'] == SELECTED_CITY]
        in_total_traffic = in_total_traffic.groupby('Year')['AdjustedForecastedDemand'].aggregate('sum').reset_index(drop = False)
        out_total_traffic = expected_route_traffic_df[expected_route_traffic_df['From'] == SELECTED_CITY]
        out_total_traffic = out_total_traffic.groupby('Year')['AdjustedForecastedDemand'].aggregate('sum').reset_index(drop = False)
        in_out_total_traffic = pd.merge(in_total_traffic, out_total_traffic, on = 'Year', suffixes = ('_InTraffic', '_OutTraffic'))

        # Add some adjustment to the forecasts based on difference between current year's actual & forecasted air travel demand
        actual_present_out_traffic = valid_route_traffic_df[(valid_route_traffic_df['From'] == SELECTED_CITY) & (valid_route_traffic_df['To'] == selected_hub_city)]
        if(actual_present_out_traffic.shape[0] > 0):    # If actual demand for route exists, make adjustments
            assert(actual_present_out_traffic.shape[0] == 1)
            actual_present_out_traffic = actual_present_out_traffic.iloc[0]['Passengers_Target'] * target_std + target_mean
            out_traffic_adjustment = actual_present_out_traffic - in_out_total_traffic.iloc[0]['AdjustedForecastedDemand_OutTraffic']
            in_out_total_traffic['AdjustedForecastedDemand_OutTraffic'] = in_out_total_traffic['AdjustedForecastedDemand_OutTraffic'] + out_traffic_adjustment
        actual_present_in_traffic = valid_route_traffic_df[(valid_route_traffic_df['To'] == SELECTED_CITY) & (valid_route_traffic_df['From'] == selected_hub_city)]
        if(actual_present_in_traffic.shape[0] > 0):    # If actual demand for route exists, make adjustments
            actual_present_in_traffic = actual_present_in_traffic.iloc[0]['Passengers_Target'] * target_std + target_mean
            in_traffic_adjustment = actual_present_in_traffic - in_out_total_traffic.iloc[0]['AdjustedForecastedDemand_InTraffic']
            in_out_total_traffic['AdjustedForecastedDemand_InTraffic'] = in_out_total_traffic['AdjustedForecastedDemand_InTraffic'] + in_traffic_adjustment

        # Return traffic forecasts along with other info about route
        in_out_total_traffic.to_csv(f"{output_save_path}/Forecasted_Route_Demand/City{SELECTED_CITY}_Hub{SELECTED_HUB_AIRPORT}.csv", index = None)
        return in_out_total_traffic, (duration_in, duration_out, railway_num, railway_duration, railway_capacity)

    airport_to_city_mapping = dict(zip(preprocessor.city_mapping['AirRouteData_AirportCode'].values, preprocessor.city_mapping['City'].values))
    city_to_airport_mapping = dict(zip(preprocessor.city_mapping['City'].values, preprocessor.city_mapping['AirRouteData_AirportCode'].values))

    # Clear previously saved forecasts to save the new forecasts
    shutil.rmtree(f"{output_save_path}/Forecasted_Route_Demand/", ignore_errors = True)
    os.mkdir(f"{output_save_path}/Forecasted_Route_Demand/")
    route_info_df = []
    city = selected_city
    route_forecasted_demands_dict = {}

    # ONLY_HUBS means whether to make connections only to hubs of network or any airport in network
    if(ONLY_HUBS == True):
        uniq_hubs = preprocessor.network_data[preprocessor.network_data['FromHub'] == 1]['From'].unique()
    else:
        uniq_hubs = pd.concat([preprocessor.network_data['From'], preprocessor.network_data['To']]).unique()
        uniq_hubs = [x for x in uniq_hubs if x in airport_to_city_mapping]
    
    for hub in uniq_hubs:
        SELECTED_CITY = city
        SELECTED_HUB_AIRPORT = hub
        route_forecasted_demands, durations_railway = get_route_demand_forecasts(SELECTED_CITY, SELECTED_HUB_AIRPORT)    # Get forecasts for given route
        assert(route_forecasted_demands.iloc[0]['Year'] == PRESENT_YEAR)
        assert(route_forecasted_demands.iloc[route_forecasted_demands.shape[0] - 1]['Year'] == FORECAST_YEAR)
        route_forecasted_demands_dict[f'{SELECTED_CITY}-{SELECTED_HUB_AIRPORT}'] = route_forecasted_demands

        # Extracting market info - number of existing flights & minimum fares
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

        # Extracting distance of flight
        DISTANCE = (get_timing(selected_city_airport, SELECTED_HUB_AIRPORT)[1] + get_timing(SELECTED_HUB_AIRPORT, selected_city_airport)[1]) / 2.0

        # Collecting all information about route
        route_info_df.append([
            SELECTED_CITY, SELECTED_HUB_AIRPORT,
            durations_railway[0], durations_railway[1],
            durations_railway[2], durations_railway[3], durations_railway[4],
            route_forecasted_demands.iloc[1]['AdjustedForecastedDemand_InTraffic'],
            route_forecasted_demands.iloc[1]['AdjustedForecastedDemand_OutTraffic'],
            route_forecasted_demands.iloc[route_forecasted_demands.shape[0] - 1]['AdjustedForecastedDemand_InTraffic'],
            route_forecasted_demands.iloc[route_forecasted_demands.shape[0] - 1]['AdjustedForecastedDemand_OutTraffic'],
            NUM_OUT_MARKET, NUM_IN_MARKET,
            PRICE_OUT_MARKET, PRICE_IN_MARKET,
            DISTANCE
        ])
    
    # Collecting all routes' information
    route_info_df = pd.DataFrame(route_info_df, columns = [
        'City', 'Hub',
        'IncomingFlightDuration', 'OutgoingFlightDuration',
        'RailwayNum', 'RailwayDuration', 'RailwayCapacity',
        'PresentYearInForecast', 'PresentYearOutForecast',
        'ForecastYearInForecast', 'ForecastYearOutForecast',
        'NUMBER_PLANES_OUT_MARKET', 'NUMBER_PLANES_IN_MARKET',
        'PRICE_OUT_MARKET', 'PRICE_IN_MARKET',
        'DISTANCE'
    ])

    # Calculating growth in demand for route
    route_info_df['GrowthIn'] = (route_info_df['ForecastYearInForecast'] - route_info_df['PresentYearInForecast']) / (route_info_df['PresentYearInForecast'] + 1e-12) * 100 / (FORECAST_YEAR - PRESENT_YEAR)
    route_info_df['GrowthOut'] = (route_info_df['ForecastYearOutForecast'] - route_info_df['PresentYearOutForecast']) / (route_info_df['PresentYearOutForecast'] + 1e-12) * 100 / (FORECAST_YEAR - PRESENT_YEAR)
    route_info_df['AvgGrowth'] = (route_info_df['GrowthIn'] + route_info_df['GrowthOut']) / 2.0

    # We sort routes based on both forecasted demand & growth rate
    # Top routes will be ones having high forecasted demand & also high growth rate in future
    route_info_df['SortingCriteria'] = route_info_df.apply(lambda x: (x['ForecastYearOutForecast'] - np.mean(route_info_df['ForecastYearOutForecast'])) / (np.std(route_info_df['ForecastYearOutForecast']) + 1e-12) + (x['AvgGrowth'] - np.mean(route_info_df['AvgGrowth'])) / (np.std(route_info_df['AvgGrowth']) + 1e-12), axis = 1)
    route_info_df = route_info_df.sort_values('SortingCriteria', ascending = False)
    route_info_df = route_info_df.drop('SortingCriteria', axis = 1)
    route_info_df['Route'] = route_info_df.apply(lambda x: x['City'] + '-' + x['Hub'], axis = 1)
    
    # Out of top 10 cities, randomly select 5 (but in order) to show on dashboard
    route_info_df = route_info_df.head(10)
    route_info_df = route_info_df[route_info_df['AvgGrowth'] > 0].reset_index(drop = True)
    random_idx = np.random.choice(np.arange(route_info_df.shape[0]), 5, replace = False, p = np.arange(route_info_df.shape[0], 0, -1) / sum(np.arange(route_info_df.shape[0], 0, -1)))
    random_idx = list(sorted(random_idx))
    route_info_df = route_info_df.loc[random_idx].reset_index(drop = True)
    route_info_df = route_info_df.set_index('Route')

    # Plot graphs
    route_forecasted_demands_dict = dict([(x, route_forecasted_demands_dict[x]) for x in route_forecasted_demands_dict if x in [*route_info_df.index]])
    plotly_RouteSelection([*route_info_df.index], route_forecasted_demands_dict, plotly_save_path)

    route_info_df = OrderedDict(route_info_df.to_dict(orient = 'index'))    
    return route_info_df

# If name is too long, we shorten it to not disturb web UI
def shorten_name(name):
    if(len(name) > 12):
        return ''.join(name[:10]) + '...'
    else:
        return name

# Function to plot graphs for Route Selection stage
def plotly_RouteSelection(routes, demand_forecasts, plotly_save_path):
    
    for route in routes:    # For each of the 5 routes which will be displayed on website

        # Graph is to visualize air travel demand forecasts for to & fro route
        city, hub = route.split('-')
        route_in_demand = demand_forecasts[route]['AdjustedForecastedDemand_InTraffic'].values[1:]
        route_out_demand = demand_forecasts[route]['AdjustedForecastedDemand_OutTraffic'].values[1:]
        years = demand_forecasts[route]['Year'].values[1:]
        
        fig1 = make_subplots(
            rows = 2, cols = 1,
            subplot_titles = [f"{shorten_name(city)}→{hub}", f"{hub}→{shorten_name(city)}"],
            shared_xaxes = True
        )
        
        fig1.add_trace(
            go.Bar(
                x = years, y = route_out_demand,
                hovertext = [f"Year: {x}<br>Passenger Forecast: {int(y)}" for x, y in zip(years, route_out_demand)],
                hoverinfo = 'text', marker = dict(color = '#2C88D9')
            ),
            row = 1, col = 1
        )
        
        fig1.add_trace(
            go.Bar(
                x = years, y = route_in_demand,
                hovertext = [f"Year: {x}<br>Passenger Forecast: {int(y)}" for x, y in zip(years, route_in_demand)],
                hoverinfo = 'text', marker = dict(color = '#2C88D9')
            ),
            row = 2, col = 1
        )
        
        fig1.update_layout(
            title_text = f"Forecasted Total Air-traffic Demand",
            height = 700, width = 500,
            paper_bgcolor = '#DBD8FD' , plot_bgcolor = '#DBD8FD',
            titlefont = dict(size = 20),
            showlegend = False
        )
        
        div1 = pyo.plot(fig1, output_type = 'div', include_plotlyjs = False, show_link = False, link_text = "", config = {"displayModeBar": False, "showTips": False})
        with open(f'{plotly_save_path}/{route}_RouteSelection_Graph1.txt', 'w') as save_file:
            save_file.write(div1)