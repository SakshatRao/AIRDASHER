import numpy as np
import pandas as pd

import geopy.distance

import shutil
import os

import time
import json

import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

from collections import OrderedDict

# Script for selecting city with highest growth potential
def NewAirport_Script(general_params, preprocessor, tier_1_2_cities_raw, output_save_path, plotly_save_path):

    PRESENT_YEAR = general_params['PRESENT_YEAR']
    FORECAST_YEAR = general_params['FORECAST_YEAR']

    city_to_airport_map = dict(zip(preprocessor.city_mapping['City'].values, preprocessor.city_mapping['AirRouteData_AirportCode']))
    tier_1_2_cities = [x for x in tier_1_2_cities_raw if pd.isnull(city_to_airport_map[x]) == True]

    def get_economic_data(cities):
        city_to_district_economic_data_mapping = dict(zip(preprocessor.city_mapping['City'], preprocessor.city_mapping['EconomicData_District']))
        cities_latest_gdps = []
        cities_gdp_history = []
        cities_latest_years = []
        for city in cities:
            district = city_to_district_economic_data_mapping[city]
            if(pd.isnull(district)):
                cities_latest_gdps.append(np.nan)
                cities_gdp_history.append([])
                cities_latest_years.append(np.nan)
            else:
                if(district.startswith('{')):
                    districts = district[1:-1]
                    districts = districts.split(' + ')
                    has_district = pd.Series([False] * preprocessor.economic_data.shape[0])
                    for district in districts:
                        has_district = has_district | (preprocessor.economic_data['District'] == district)
                    city_economic_data = preprocessor.economic_data[has_district]
                    city_economic_data = city_economic_data.groupby(['Year'])[['GDP']].aggregate('sum').reset_index(drop = False).rename({'index': 'Year'})
                    city_economic_data['State'] = pd.Series([''] * city_economic_data.shape[0])
                    city_economic_data['District'] = pd.Series([''] * city_economic_data.shape[0])
                else:
                    if(district == 'BILASPUR'):
                        city_economic_data = preprocessor.economic_data[(preprocessor.economic_data['District'] == district) & (preprocessor.economic_data['State'] == 'Chattisgarh')]
                    elif(district == 'AURANGABAD'):
                        city_economic_data = preprocessor.economic_data[(preprocessor.economic_data['District'] == district) & (preprocessor.economic_data['State'] == 'Maharashtra')]
                    elif(district == 'HAMIRPUR'):
                        city_economic_data = preprocessor.economic_data[(preprocessor.economic_data['District'] == district) & (preprocessor.economic_data['State'] == 'HimachalPradesh')]
                    else:
                        city_economic_data = preprocessor.economic_data[preprocessor.economic_data['District'] == district]
                city_economic_data = city_economic_data.sort_values("Year").reset_index(drop = True)
                city_economic_data['Year'] = city_economic_data['Year'].apply(lambda x: int(x.split('-')[0]))
                if(city_economic_data.iloc[city_economic_data.shape[0] - 1]['Year'] - city_economic_data.iloc[0]['Year'] != city_economic_data.shape[0] + 1):
                    start_year = city_economic_data.iloc[0]['Year']; last_year = city_economic_data.iloc[city_economic_data.shape[0] - 1]['Year']
                    for year in range(start_year + 1, last_year):
                        if(year not in [*city_economic_data['Year'].values]):
                            to_add_row = city_economic_data.iloc[0]
                            to_add_row['Year'] = year
                            to_add_row['GDP'] = np.nan
                            to_add_row = pd.DataFrame([to_add_row], columns = city_economic_data.columns)
                            city_economic_data = pd.concat([city_economic_data, to_add_row], axis = 0).reset_index(drop = True)
                city_economic_data = city_economic_data.sort_values("Year").reset_index(drop = True)
                try:
                    assert(city_economic_data.iloc[city_economic_data.shape[0] - 1]['Year'] - city_economic_data.iloc[0]['Year'] == city_economic_data.shape[0] - 1)
                except:
                    print(city_economic_data)
                    print(city)
                    exit()
                city_latest_gdp = city_economic_data.iloc[city_economic_data.shape[0] - 1]['GDP']
                city_latest_year = int(city_economic_data.iloc[city_economic_data.shape[0] - 1]['Year'])
                cities_latest_gdps.append(city_latest_gdp)
                city_gdp_history = city_economic_data['GDP'].values
                cities_gdp_history.append(city_gdp_history)
                cities_latest_years.append(city_latest_year)
        economic_data = pd.DataFrame([cities, cities_latest_gdps, cities_gdp_history, cities_latest_years], index = ['City', 'GDP_economic_latest', 'GDP_economic_1y_history', 'GDP_economic_1y_latestyear']).transpose().reset_index(drop = True)
        for col in economic_data.columns[1:-2]:
            economic_data[col] = economic_data[col].fillna(np.nanmean(economic_data[col].values))
        return economic_data

    economic_data = get_economic_data(tier_1_2_cities)

    def get_tourism_data(cities):

        def distance_factor(miles):
            return np.exp(miles / (-200))

        def closest_node(node, nodes):
            dist = np.sum((nodes - node)**2, axis=1)
            return np.argmin(dist)

        all_cities_coords = preprocessor.city_mapping[preprocessor.city_mapping['City'].isin(cities)]
        all_cities_coords = all_cities_coords['Airport_City_Coords']
        all_cities_coords_lat = all_cities_coords.apply(lambda x: float(x.split(',')[0])).values
        all_cities_coords_lon = all_cities_coords.apply(lambda x: float(x.split(',')[1])).values
        all_cities_coords = np.asarray(list(zip(all_cities_coords_lat, all_cities_coords_lon)))

        all_cities_tourism_dict = {}
        for city in cities:
            all_cities_tourism_dict[city] = np.zeros(preprocessor.monument_visitors_data.shape[1] - 1, dtype = 'float')

        for idx, row in preprocessor.tourist_loc_coords_data.iterrows():
            tourist_loc_coord = np.asarray([row['Latitude'], row['Longitude']])
            closest_idx = closest_node(tourist_loc_coord, all_cities_coords)
            closest_city = cities[closest_idx]
            closest_coord = all_cities_coords[closest_idx]
            closest_dist = geopy.distance.geodesic(tourist_loc_coord, closest_coord).miles
            factor = distance_factor(closest_dist)
            tourism_data = preprocessor.monument_visitors_data.loc[idx].values
            if(closest_city in all_cities_tourism_dict):
                all_cities_tourism_dict[closest_city] += np.asarray(tourism_data[1:] * factor, dtype = 'float64')
            else:
                all_cities_tourism_dict[closest_city] = np.asarray(tourism_data[1:] * factor, dtype = 'float64')
        tourism_data = pd.DataFrame.from_dict(all_cities_tourism_dict, orient = 'index', columns = preprocessor.monument_visitors_data.columns[1:]).reset_index(drop = False).rename({'index': 'City'}, axis = 1)
        tourism_data['Domestic_tourism_1y_history'] = tourism_data.apply(lambda x: [x['Domestic2018-19'], x['Domestic2019-20'], x['Domestic2020-21'], x['Domestic2021-22']], axis = 1)
        tourism_data['Domestic_tourism_1y_latestyear'] = pd.Series([2021] * tourism_data.shape[0])
        tourism_data['Foreign_tourism_1y_history'] = tourism_data.apply(lambda x: [x['Foreign2018-19'], x['Foreign2019-20'], x['Foreign2020-21'], x['Foreign2021-22']], axis = 1)
        tourism_data['Foreign_tourism_1y_latestyear'] = pd.Series([2021] * tourism_data.shape[0])
        tourism_data.drop(['Domestic2018-19', 'Domestic2019-20', 'Domestic2020-21', 'Foreign2018-19', 'Foreign2019-20', 'Foreign2020-21', 'NumMonuments2020'], axis = 1, inplace = True)
        tourism_data = tourism_data.rename({'Domestic2021-22': 'Domestic_tourism_latest', 'Foreign2021-22': 'Foreign_tourism_latest', 'NumMonuments2022': 'NumMonuments_tourism_latest'}, axis = 1)
        return tourism_data

    tourism_data = get_tourism_data(tier_1_2_cities)

    def get_education_data(cities):
        city_to_district_education_data_mapping = dict(zip(preprocessor.city_mapping['City'], preprocessor.city_mapping['EducationData_District']))
        cities_education_data = []
        for city in cities:
            city_to_district_map = city_to_district_education_data_mapping[city]
            district_education_history = np.zeros((preprocessor.education_data.shape[1] - 3, 3))
            if(city_to_district_map.startswith('{')):
                city_to_district_map = city_to_district_map[1:-1]
                districts_years = []
                if('1991' in city_to_district_map):
                    dict_entries = city_to_district_map.split('\n')
                    for entry in dict_entries:
                        year = int(entry.split(': ')[0])
                        entry = ': '.join(entry.split(': ')[1:])
                        if('State' in entry):
                            district = ""
                            state = entry.split('State: ')[1]
                        else:
                            district = entry
                            state = ""
                        districts_years.append([district, year, state])
                else:
                    if('+' in city_to_district_map):
                        districts = city_to_district_map.split(' + ')
                        for district in districts:
                            districts_years.extend([(district, 1991, ''), (district, 2001, ''), (district, 2011, '')])
            else:
                district = city_to_district_map
                districts_years = [(district, 1991, ''), (district, 2001, ''), (district, 2011, '')]
            history = []
            years = [1991, 2001, 2011]
            for year_idx, (district, year, state) in enumerate(districts_years):
                if(state == ''):
                    if(district == 'BILASPUR'):
                        district_education_data = preprocessor.education_data[(preprocessor.education_data['District'] == district) & (preprocessor.education_data['State'] == 'CHHATTISGARH')]
                    elif(district == 'AURANGABAD'):
                        district_education_data = preprocessor.education_data[(preprocessor.education_data['District'] == district) & (preprocessor.education_data['State'] == 'MAHARASHTRA')]
                    elif(district == 'HAMIRPUR'):
                        district_education_data = preprocessor.education_data[(preprocessor.education_data['District'] == district) & (preprocessor.education_data['State'] == 'HIMACHAL PRADESH')]
                    elif(district == 'BIJAPUR'):
                        district_education_data = preprocessor.education_data[(preprocessor.education_data['District'] == district) & (preprocessor.education_data['State'] == 'KARNATAKA')]
                    else:
                        district_education_data = preprocessor.education_data[preprocessor.education_data['District'] == district]
                else:
                    district_education_data = preprocessor.education_data[preprocessor.education_data['State'] == state]
                    district_education_data = district_education_data.groupby('Year')[district_education_data.columns[:-3]].aggregate('sum').reset_index(drop = False).rename({'index': 'Year'})
                    district_education_data['State'] = pd.Series([state] * district_education_data.shape[0])
                    district_education_data['District'] = pd.Series([''] * district_education_data.shape[0])
                    district_education_data = district_education_data[[*district_education_data.columns[1:]] + [district_education_data.columns[0]]]
                yearly_district_education_data = district_education_data[district_education_data['Year'] == year]
                if(yearly_district_education_data.shape[0] == 0):
                    district_education_history[:, years.index(year)] = np.nan
                    continue
                try:
                    assert(yearly_district_education_data.shape[0] == 1)
                except:
                    print(district)
                    print(state)
                    exit(1)
                for col_idx, col in enumerate(yearly_district_education_data.columns[:-3]):
                    district_education_history[col_idx, years.index(year)] += yearly_district_education_data.iloc[0][col]
            district_education_latestyear = [2011] * district_education_history.shape[0]
            district_education_history = [*district_education_history]
            district_education_data = [city, *[x[2] for x in district_education_history]]
            cities_education_data.append(district_education_data + district_education_history + district_education_latestyear)
        cities_education_data = pd.DataFrame(cities_education_data, columns = ['City'] + [x + "_education_latest" for x in preprocessor.education_data.columns[:-3]] + [x + '_education_10y_history' for x in preprocessor.education_data.columns[:-3]] + [x + '_education_10y_latestyear' for x in preprocessor.education_data.columns[:-3]])
        return cities_education_data

    education_data = get_education_data(tier_1_2_cities)

    def get_population_household_area_data(cities):
        city_to_district_pop_area_household_data_mapping = dict(zip(preprocessor.city_mapping['City'], preprocessor.city_mapping['PopulationAreaHousehold_District']))
        pop_area_household_data = pd.DataFrame()
        for city in cities:
            district = city_to_district_pop_area_household_data_mapping[city]
            if(district.startswith('{')):
                districts = district[1:-1].split(' + ')
                has_district = pd.Series([False] * preprocessor.pop_area_household_data.shape[0])
                for district in districts:
                    has_district = has_district | (preprocessor.pop_area_household_data['District'] == district)
                district_data = preprocessor.pop_area_household_data[has_district]
                district_data = district_data.drop(['StateCode', 'IsDistrict', 'District', 'IsTotal'], axis = 1)
                district_data = pd.DataFrame([district_data.sum(axis = 0)], columns = district_data.columns)
                district_data['StateCode'] = pd.Series([''] * district_data.shape[0])
                district_data['IsDistrict'] = pd.Series(['DISTRICT'] * district_data.shape[0])
                district_data['District'] = pd.Series([''] * district_data.shape[0])
                district_data['IsTotal'] = pd.Series(['Total'] * district_data.shape[0])
                district_data = district_data[[*district_data.columns[-4:]] + [*district_data.columns[:-4]]]
            else:
                if(district == 'Bilaspur'):
                    district_data = preprocessor.pop_area_household_data[(preprocessor.pop_area_household_data['District'] == district) & (preprocessor.pop_area_household_data['StateCode'] == 22)]
                elif(district == 'Hamirpur'):
                    district_data = preprocessor.pop_area_household_data[(preprocessor.pop_area_household_data['District'] == district) & (preprocessor.pop_area_household_data['StateCode'] == 2)]
                elif(district == 'Bijapur'):
                    district_data = preprocessor.pop_area_household_data[(preprocessor.pop_area_household_data['District'] == district) & (preprocessor.pop_area_household_data['StateCode'] == 29)]
                else:
                    district_data = preprocessor.pop_area_household_data[preprocessor.pop_area_household_data['District'] == district]
            try:
                assert(district_data.shape[0] == 1)
            except:
                print(district_data.shape)
                print(district)
                exit(1)
            district_data = pd.DataFrame(district_data.values[:, 4:], columns = [x + '_population_latest' for x in district_data.columns[4:]])
            district_data['City'] = pd.Series([city])
            pop_area_household_data = pd.concat([pop_area_household_data, district_data], axis = 0)
        pop_area_household_data = pop_area_household_data[[*pop_area_household_data.columns[-1:]] + [*pop_area_household_data.columns[:-1]]]
        pop_area_household_data['PopulationPerSqKm_population_latest'] = pop_area_household_data['Population_population_latest'] / pop_area_household_data['Area_population_latest']
        return pop_area_household_data

    pop_household_area_data = get_population_household_area_data(tier_1_2_cities)

    def get_latest_population_data(cities):
        city_to_city_latest_population_data_mapping = dict(zip(preprocessor.city_mapping['City'], preprocessor.city_mapping['LatestPopulation_City']))
        latest_population_data = pd.DataFrame()
        for city in cities:
            mapped_city = city_to_city_latest_population_data_mapping[city]
            if(pd.isnull(mapped_city)):
                city_data = pd.DataFrame.from_dict({'City': [city], 'pop2023': [np.nan]}, orient = 'columns')
            elif(mapped_city.startswith('{')):
                mapped_cities = mapped_city[1:-1].split(' + ')
                has_city = pd.Series([False] * preprocessor.latest_population_data.shape[0])
                for mapped_city in mapped_cities:
                    has_city = has_city | (preprocessor.latest_population_data['city'] == mapped_city)
                city_data = preprocessor.latest_population_data[has_city]
                city_data = city_data.drop(['latitude', 'longitude', 'city'], axis = 1)
                city_data = pd.DataFrame([city_data.sum(axis = 0)], columns = city_data.columns)
                city_data['City'] = pd.Series([''] * city_data.shape[0])
                city_data = city_data[[*city_data.columns[-1:]] + [*city_data.columns[:-1]]]
            else:
                city_data = preprocessor.latest_population_data[preprocessor.latest_population_data['city'] == mapped_city]
                city_data = city_data.drop(['latitude', 'longitude'], axis = 1)
            assert(city_data.shape[0] == 1)
            city_data = pd.DataFrame(city_data.values[:, 1:], columns = [x + '_population_latest' for x in city_data.columns[1:]])
            city_data['City'] = pd.Series([city])
            latest_population_data = pd.concat([latest_population_data, city_data], axis = 0)
        latest_population_data = latest_population_data[[*latest_population_data.columns[-1:]] + [*latest_population_data.columns[:-1]]]
        return latest_population_data

    latest_population_data = get_latest_population_data(tier_1_2_cities)

    def get_population_history_data(cities):
        city_to_district_pop_history_data_mapping = dict(zip(preprocessor.city_mapping['City'], preprocessor.city_mapping['PopulationHistory_District']))
        pop_history_data = pd.DataFrame()
        population_history_data = preprocessor.population_history_data.copy()
        for city in cities:
            district = city_to_district_pop_history_data_mapping[city]
            district_population_history_data = np.zeros((3, 12))
            if(district.startswith('{')):
                districts = district[1:-1].split(' + ')
                district_data = population_history_data[districts[0]]['history']
                for district in districts[1:]:
                    next_district_data = preprocessor.population_history_data[district]['history']
                    for year in np.arange(1901, 2012, 10):
                        if((str(int(year)) not in district_data) & (str(int(year)) in next_district_data)):
                            district_data[str(int(year))] = next_district_data[str(int(year))]
                        elif((str(int(year)) in district_data) & (str(int(year)) in next_district_data)):
                            for pop_type_idx in range(3):
                                district_data[str(int(year))][pop_type_idx] = district_data[str(int(year))][pop_type_idx] + next_district_data[str(int(year))][pop_type_idx]
            else:
                district_data = population_history_data[district]['history']
            for year_idx, year in enumerate(np.arange(1901, 2012, 10)):
                if(str(int(year)) in district_data):
                    for pop_type_idx in range(3):
                        if((type(district_data[str(int(year))][pop_type_idx]) == str)):
                            if((district_data[str(int(year))][pop_type_idx].strip().startswith('N')) | (district_data[str(int(year))][pop_type_idx].strip().startswith('-'))):
                                district_population_history_data[pop_type_idx, year_idx] = np.nan
                            else:
                                print("ALERT!")
                                print(district_data[str(int(year))][pop_type_idx])
                        else:
                            district_population_history_data[pop_type_idx, year_idx] = district_data[str(int(year))][pop_type_idx]
                else:
                    district_population_history_data[:, year_idx] = [np.nan] * 3
            district_population_history_data = pd.DataFrame([[list(x) for x in district_population_history_data]], columns = ['Population_population_10y_history', 'MalePopulation_population_10y_history', 'FemalePopulation_population_10y_history'])
            district_population_history_data['City'] = pd.Series([city])
            pop_history_data = pd.concat([pop_history_data, district_population_history_data], axis = 0)
        pop_history_data = pop_history_data[[*pop_history_data.columns[-1:]] + [*pop_history_data.columns[:-1]]]
        pop_history_data['Population_population_10y_latestyear'] = pd.Series([2011] * pop_history_data.shape[0])
        pop_history_data['MalePopulation_population_10y_latestyear'] = pd.Series([2011] * pop_history_data.shape[0])
        pop_history_data['FemalePopulation_population_10y_latestyear'] = pd.Series([2011] * pop_history_data.shape[0])
        return pop_history_data

    pop_history_data = get_population_history_data(tier_1_2_cities)

    def get_airport_in_out_passenger_traffic_data(cities):
        city_to_city_mapping = dict(zip(preprocessor.city_mapping['City'], preprocessor.city_mapping['DomesticPassengerTraffic_City']))
        airport_in_out_traffic = []
        for city in cities:
            airport = city_to_city_mapping[city]
            if(pd.isnull(airport) == True):
                airport_in_out_traffic.append([city, np.nan])
            else:
                airport_flights = preprocessor.total_domestic_data[(preprocessor.total_domestic_data['FROM'] == airport) | (preprocessor.total_domestic_data['TO'] == airport)]
                in_out_traffic = airport_flights['PASSENGERS'].sum()
                airport_in_out_traffic.append([city, in_out_traffic])
        airport_in_out_traffic_data = pd.DataFrame(airport_in_out_traffic, columns = ['City', 'In_Out_Traffic_target'])
        return airport_in_out_traffic_data

    airport_traffic_data = get_airport_in_out_passenger_traffic_data(tier_1_2_cities)

    # Collecting all features & targets together
    all_datasets_list = [economic_data, tourism_data, education_data, pop_household_area_data, latest_population_data, pop_history_data, airport_traffic_data]
    total_dataset = all_datasets_list[0]
    for dataset in all_datasets_list[1:]:
        total_dataset = pd.merge(total_dataset, dataset, on = 'City')

    latest_features = [x for x in total_dataset.columns if x.endswith('_latest')]
    history_features = [x for x in total_dataset.columns if x.endswith('_history')]
    latestyear_features = [x for x in total_dataset.columns if x.endswith('_latestyear')]
    target_feature = [x for x in total_dataset.columns if x.endswith('_target')]
    assert(len(latest_features) + len(history_features) + len(target_feature) + len(latestyear_features) + 1 == total_dataset.shape[1])

    N_COMPONENTS = 2
    categories = ['economic', 'tourism', 'education', 'population']
    categories_cols = []
    for category in categories:
        categories_cols.extend([f"{category}_pca{n}" for n in range(1, N_COMPONENTS + 1)])
    
    cols_standardization_vals = preprocessor.CitySelection_cols_standardization_vals

    class LinearModel:
        def __init__(self, coefs):
            self.coefs = coefs
        def predict(self, X):
            assert(len(self.coefs) == X.shape[1] + 1)
            return np.dot([*self.coefs.values()][1:], X.transpose()) + self.coefs['intercept']

    model = LinearModel(preprocessor.CitySelection_model_coefs)

    def get_forecasted_values(total_dataset_raw, FORECAST_YEAR, get_plot_info = False):
        total_dataset = total_dataset_raw.copy()
        history_to_latest_feature_mapping = dict(zip(history_features, ['_'.join(x.split('_history')[0].split('_')[:-1]) + '_latest' for x in history_features if '_'.join(x.split('_history')[0].split('_')[:-1]) + '_latest' in latest_features]))
        growth = np.zeros((total_dataset.shape[0], len(history_features)))
        forecasts = np.zeros((total_dataset.shape[0], len(history_features)))
        all_plot_info = {}
        for feature_idx, feature in enumerate(history_features):
            for idx, row in total_dataset.iterrows():
                city_history_feature = row[feature]
                city_history_latestyear = row[feature.split('_history')[0] + '_latestyear']
                if(len(city_history_feature) != 0):
                    x = np.arange(len(city_history_feature))
                    y = np.asarray(city_history_feature) + 1    # To avoid log(0)
                    non_na_idx = (pd.isnull(y) == False)
                    x = x[non_na_idx]
                    y = y[non_na_idx]
                    def remove_outliers(y_raw):
                        y = np.cumsum(y_raw, dtype=float) / np.arange(1, len(y_raw) + 1)
                        q1 = np.quantile(y, 0.25)
                        q3 = np.quantile(y, 0.75)
                        iqr = q3 - q1
                        return [x for x in range(len(y)) if ((y[x] >= (q1 - 1.5 * iqr)) & (y[x] <= (q3 + 1.5 * iqr)))]
                    non_outlier_idx = remove_outliers(y)
                    x = x[non_outlier_idx]
                    y = y[non_outlier_idx]
                    if(len(x) > 1):
                        y = np.log(y)
                        curve_fit = np.polyfit(x, y, 1)
                        growth[idx, feature_idx] = curve_fit[0]
                        if(feature.split('_')[-2] == '1y'):
                            duration = 1
                        elif(feature.split('_')[-2] == '10y'):
                            duration = 10
                        else:
                            print("ALERT: Invalid duration provided!")
                        forecast_idx = (FORECAST_YEAR - city_history_latestyear) / duration + len(city_history_feature) - 1
                        forecast = np.exp(curve_fit[0] * forecast_idx + curve_fit[1]) - 1
                        forecasts[idx, feature_idx] = forecast
                        # For visualizing the growths
                        if((feature in ['GDP_economic_1y_history', 'Domestic_tourism_1y_history', '25-29_Graduates_education_10y_history', 'Population_population_10y_history']) and (get_plot_info == True)):
                            long_x = [*x] + [*np.arange(x[-1] + 1, int(forecast_idx) + 1)]
                            if(long_x[-1] != forecast_idx):
                                long_x.append(forecast_idx)
                            long_x = np.asarray(long_x)
                            all_years = [(val + 1 - len(city_history_feature)) * duration + city_history_latestyear for val in long_x]
                            x_years = [(val + 1 - len(city_history_feature)) * duration + city_history_latestyear for val in x]
                            fit_y = np.exp(curve_fit[0] * long_x + curve_fit[1])

                            plot_info = {}
                            plot_info['x_years'] = x_years
                            plot_info['y'] = y
                            plot_info['all_years'] = all_years
                            plot_info['fit_y'] = fit_y
                            plot_info['forecast'] = forecast
                            plot_info['growth'] = curve_fit[0]
                            plot_info['duration'] = duration

                            all_plot_info[f'{row["City"]}_{feature}'] = plot_info
                    else:
                        growth[idx, feature_idx] = np.nan
                        forecasts[idx, feature_idx] = y[0]
                else:
                    growth[idx, feature_idx] = np.nan
                    forecasts[idx, feature_idx] = row[history_to_latest_feature_mapping[feature]]
        
        growth_df = pd.DataFrame(growth, columns = ['_'.join(x.split('_history')[0].split('_')[:-1]) + '_growth' for x in history_features])
        growth_df = growth_df.fillna(0)
        forecasts_df = pd.DataFrame(forecasts, columns = ['_'.join(x.split('_history')[0].split('_')[:-1]) + '_forecast' for x in history_features])
        growth_forecasts_df = pd.concat([growth_df, forecasts_df], axis = 1)
        growth_forecasts_df['City'] = total_dataset['City']
        growth_forecasts_df = growth_forecasts_df.reset_index(drop = True)
        total_dataset = pd.merge(total_dataset, growth_forecasts_df, on = 'City')

        forecast_features = [x for x in total_dataset.columns if x.endswith('_forecast')]
        non_forecast_col_growths = {
            'NumMonuments_tourism_latest': 0,
            'NumMonumentsChange_tourism_latest': 0,
            'InhabitedVillages_population_latest': 'Population_population',
            'UninhabitedVillages_population_latest': 'Population_population',
            'Towns_population_latest': 'Population_population',
            'Households_population_latest': 'Population_population',
            'Area_population_latest': 0,
            'PopulationPerSqKm_population_latest': 'Population_population',
            'pop2023_population_latest': 'Population_population'
        }
        for col in latest_features:
            if(col.split('_latest')[0] + '_forecast' in forecast_features):
                total_dataset[col + '2'] = total_dataset[col.split('_latest')[0] + '_forecast']
            else:
                assert(col in non_forecast_col_growths)
                if(type(non_forecast_col_growths[col]) == str):
                    total_dataset[col + '_growth'] = total_dataset[non_forecast_col_growths[col] + '_growth']
                    duration_col = [x for x in total_dataset.columns if (x.startswith(non_forecast_col_growths[col])) & (x.endswith('_history'))]
                    assert(len(duration_col) == 1)
                    if('1y' in duration_col[0]):
                        duration = 1
                    elif('10y' in duration_col[0]):
                        duration = 10
                    else:
                        print("ALERT! - Invalid duration found")
                    # (y2 + 1) = (y1 + 1) * e ^ ( (x2 - x1) * a )
                    total_dataset[col + '2'] = total_dataset.apply(lambda x: (x[col] + 1) * np.exp(x[col + '_growth'] * (FORECAST_YEAR - PRESENT_YEAR) / duration) - 1, axis = 1)
                    total_dataset = total_dataset.drop(col + '_growth', axis = 1)
                else:
                    total_dataset[col + '2'] = total_dataset[col]

        new_total_valid_data = total_dataset[['City'] + [x for x in total_dataset.columns if x.endswith('_latest2')]].copy()
        new_latest_features = [x for x in total_dataset.columns if x.endswith('_latest2')]
        for col_idx, col in enumerate(new_total_valid_data.columns):
            if(col in new_latest_features):
                col_mean = cols_standardization_vals[col[:-1]]['mean']
                col_std = cols_standardization_vals[col[:-1]]['std']
                new_total_valid_data[col] = new_total_valid_data[col].fillna(col_mean)
                new_total_valid_data[col] = (new_total_valid_data[col] - col_mean) / (col_std + 1e-20)

        new_data_X = new_total_valid_data[new_latest_features].values
        new_data_pca_X = np.zeros((new_data_X.shape[0], len(categories) * N_COMPONENTS))
        for category_idx, category in enumerate(categories):
            to_use_cols = [x for x in range(new_data_X.shape[1]) if new_latest_features[x].endswith(f"_{category}_latest2")]
            category_data = new_data_X[:, to_use_cols]
            if(category_data.shape[1] <= N_COMPONENTS):
                new_data_pca_X[:, category_idx * N_COMPONENTS: category_idx * N_COMPONENTS + category_data.shape[1]] = category_data
            else:
                pca = preprocessor.CitySelection_pca[category_idx]
                category_pca_data = np.dot(pca, category_data.transpose()).transpose()
                new_data_pca_X[:, category_idx * N_COMPONENTS: (category_idx + 1) * N_COMPONENTS] = category_pca_data
        new_data_pca_X_df = pd.DataFrame(new_data_pca_X, columns = categories_cols)
        new_data_pca_X_df['City'] = pd.Series(new_total_valid_data['City'].values)

        target_mean = cols_standardization_vals[target_feature[0]]['mean']
        target_std = cols_standardization_vals[target_feature[0]]['std']
        new_pred_traffic = model.predict(new_data_pca_X) * target_std + target_mean
        new_pred_traffic[new_pred_traffic < 0] = 0
        new_pred_traffic_df = pd.DataFrame.from_dict({'City': new_total_valid_data['City'].values, 'PredictedFutureTraffic': new_pred_traffic}, orient = 'columns')

        if(get_plot_info):
            return new_data_pca_X_df, new_pred_traffic_df, [new_total_valid_data, total_dataset], all_plot_info
        else:
            return new_data_pca_X_df, new_pred_traffic_df, [new_total_valid_data, total_dataset]

    shutil.rmtree(f"{output_save_path}/Forecasted_Features/", ignore_errors = True)
    os.mkdir(f"{output_save_path}/Forecasted_Features/")
    all_pred_traffic = pd.DataFrame()
    for year in range(PRESENT_YEAR, FORECAST_YEAR + 1):
        if(year == FORECAST_YEAR):
            new_data_pca_X_df, new_pred_traffic_df, debug_info, plot_info = get_forecasted_values(total_dataset, year, True)
        else:
            new_data_pca_X_df, new_pred_traffic_df, debug_info = get_forecasted_values(total_dataset, year, False)
        pred_traffic_df = new_pred_traffic_df.copy()
        pred_traffic_df['Year'] = pd.Series(np.repeat(year, pred_traffic_df.shape[0]))
        all_pred_traffic = pd.concat([all_pred_traffic, pred_traffic_df], axis = 0)
        if(year == PRESENT_YEAR):
            present_year_forecasts = new_pred_traffic_df
        if(year == FORECAST_YEAR):
            forecasted_traffic_df =  new_pred_traffic_df

    airport_current_traffic_df = pd.merge(airport_traffic_data, present_year_forecasts, on = 'City')
    def get_current_traffic(row):
        return row['PredictedFutureTraffic']
    airport_current_traffic_df['CurrentTraffic'] = airport_current_traffic_df.apply(get_current_traffic, axis = 1)
    airport_current_traffic_df = airport_current_traffic_df[['City', 'CurrentTraffic']]

    all_traffic_df = pd.merge(airport_current_traffic_df, forecasted_traffic_df, on = 'City')
    all_traffic_df['GrowthRate'] = all_traffic_df['PredictedFutureTraffic']

    most_growth_cities = all_traffic_df.sort_values("GrowthRate", ascending = False)
    most_growth_cities['Latitude_Longitude'] = most_growth_cities['City'].map(dict(zip(preprocessor.city_mapping['City'].values, preprocessor.city_mapping['Airport_City_Coords'].values)))
    most_growth_cities['Latitude'] = most_growth_cities['Latitude_Longitude'].apply(lambda x: float(x.split(', ')[0].strip()))
    most_growth_cities['Longitude'] = most_growth_cities['Latitude_Longitude'].apply(lambda x: float(x.split(', ')[1].strip()))
    most_growth_cities = most_growth_cities.drop('Latitude_Longitude', axis = 1)

    most_growth_cities = most_growth_cities.head(20)
    most_growth_cities = most_growth_cities[most_growth_cities['GrowthRate'] > 0].reset_index(drop = True)
    random_idx = np.random.choice(np.arange(most_growth_cities.shape[0]), 5, replace = False, p = np.arange(most_growth_cities.shape[0], 0, -1) / sum(np.arange(most_growth_cities.shape[0], 0, -1)))
    random_idx = list(sorted(random_idx))
    most_growth_cities = most_growth_cities.loc[random_idx].reset_index(drop = True)
    
    most_growth_cities_names = most_growth_cities['City'].values
    plot_info_keys = [[x for x in plot_info if x.startswith(y)] for y in most_growth_cities_names]
    plot_info_all_keys = []
    for x in plot_info_keys:
        plot_info_all_keys.extend(x)
    sel_pred_traffic = all_pred_traffic[all_pred_traffic['City'].isin(most_growth_cities_names)]
    plotly_NewAirports(most_growth_cities_names, [(x, plot_info[x]) for x in plot_info_all_keys], sel_pred_traffic, plotly_save_path)
    
    return OrderedDict(most_growth_cities.set_index('City').to_dict(orient = 'index'))

def shorten_name(name):
    if(len(name) > 12):
        return ''.join(name[:10]) + '...'
    else:
        return name

def plotly_NewAirports(cities, plot_info, pred_traffic, plotly_save_path):
    
    for city in cities:
        fig1 = make_subplots(
            rows = 1, cols = 1
        )

        city_demand = pred_traffic[pred_traffic['City'] == city].sort_values('Year')
        year = city_demand['Year'].values
        year_idx = np.arange(year.shape[0])
        vals = city_demand['PredictedFutureTraffic'].values
        fig1.add_trace(
            go.Bar(
                x = year, y =vals,
                hovertext = [f"Year: {x}<br>Passenger Forecast: {int(y)}" for x, y in zip(year, vals)],
                hoverinfo = 'text', marker = dict(color = '#1AAE9F')
            ),
            row = 1, col = 1
        )
        
        fig1.update_layout(
            title_text = f"Forecasted Total Air-traffic Demand for {shorten_name(city)}",
            height = 700, width = 500,
            paper_bgcolor = '#B8F4EE' , plot_bgcolor = '#B8F4EE',
            titlefont = dict(size = 20),
        )
        
        cols = ['Population_population_10y_history', 'GDP_economic_1y_history', '25-29_Graduates_education_10y_history', 'Domestic_tourism_1y_history']
        factors = ['Population', 'Economics', 'Education', 'Tourism']
        
        subplot_names = [''] * 4
        for city_col_name, col_info in plot_info:
            col_name = '_'.join(city_col_name.split('_')[1:])
    
            if(city_col_name.startswith(city)):
                if(col_info['growth'] > 0):
                    behavior = 'double'
                elif(col_info['growth'] < 0):
                    behavior = 'halve'
                else:
                    behavior = 'constant'
                if(behavior != 'constant'):
                    subplot_names[cols.index(col_name)] = f"<b>{factors[cols.index(col_name)]}</b>: Doubles every {round(np.log(2) / col_info['growth'] * col_info['duration'], 1)} yrs"
                else:
                    subplot_names[cols.index(col_name)] = f"<b>{factors[cols.index(col_name)]}</b>: Expected to remain constant"
        
        fig2 = make_subplots(
            rows = 2, cols = 2,
            vertical_spacing = 0.1,
            horizontal_spacing = 0.1,
            subplot_titles = subplot_names
        )
        
        legend_mapping_done = False
        
        for col_idx, (city_col_name, col_info) in enumerate(plot_info):
            
            col_name = '_'.join(city_col_name.split('_')[1:])
            
            if(legend_mapping_done == False):
                showlegend_dict = {'showlegend': True}
            else:
                showlegend_dict = {'showlegend': False}
    
            if(city_col_name.startswith(city)): 
            
                fig2.add_trace(
                    go.Scatter(
                        x = col_info['x_years'], y = np.exp(col_info['y']) - 1,
                        mode = 'markers', marker = dict(color = '#F7C325', size = 10),
                        hovertemplate = None, hoverinfo = "skip",
                        name = 'Historical Data', legendgroup = f"{col_idx}1", **showlegend_dict
                    ),
                    row = cols.index(col_name) // 2 + 1, col = cols.index(col_name) % 2 + 1
                )
                
                fig2.add_trace(
                    go.Scatter(
                        x = col_info['all_years'], y = col_info['fit_y'] - 1,
                        mode = 'lines', line = dict(color = '#1AAE9F'),
                        hovertemplate = None, hoverinfo = "skip",
                        name = 'Fitted Exponential Curve', legendgroup = f"{col_idx}2", **showlegend_dict
                    ),
                    row = cols.index(col_name) // 2 + 1, col = cols.index(col_name) % 2 + 1
                )
                
                fig2.add_trace(
                    go.Scatter(
                        x = [col_info['all_years'][-1]], y = [col_info['forecast']],
                        mode = 'markers', marker = dict(color = '#1AAE9F', size = 15, opacity = 0.3),
                        hovertemplate = None, hoverinfo = "skip",
                        name = 'Forecast', legendgroup = f"{col_idx}3", **showlegend_dict
                    ),
                    row = cols.index(col_name) // 2 + 1, col = cols.index(col_name) % 2 + 1
                )
                
                legend_mapping_done = True
            
        fig2.update_layout(
            title_text = f"Doubling period for various Macro-Economic Factors",
            titlefont = dict(size = 20),
            paper_bgcolor = '#B8F4EE' , plot_bgcolor = '#B8F4EE',
            height = 700, width = 700,
            legend = dict(
                orientation="h",
                yanchor="top",
                y=-0.05,
                xanchor="right",
                x=1,
                itemclick="toggleothers",
                itemdoubleclick="toggle",
            )
        )
        fig2.update_yaxes(automargin = True)
        fig2.update_xaxes(automargin = True)
        fig2.update_annotations(font_size = 15)
        
        #if(city == cities[0]):
        #   pyo.plot(fig1, output_type = 'file', filename = f'{plotly_save_path}/CityRouteSelection.html', config = {"displayModeBar": False, "showTips": False})
        #   pyo.plot(fig2, output_type = 'file', filename = f'{plotly_save_path}/{city}_CityRouteSelection2.html', config = {"displayModeBar": False, "showTips": False})
        
        div1 = pyo.plot(fig1, output_type = 'div', include_plotlyjs = False, show_link = False, link_text = "", config = {"displayModeBar": False, "showTips": False})
        with open(f'{plotly_save_path}/{city}_NewAirports_Graph1.txt', 'w') as save_file:
            save_file.write(div1)
        div2 = pyo.plot(fig2, output_type = 'div', include_plotlyjs = False, show_link = False, link_text = "", config = {"displayModeBar": False, "showTips": False})
        with open(f'{plotly_save_path}/{city}_NewAirports_Graph2.txt', 'w') as save_file:
            save_file.write(div2)

# # Testing
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
# most_growth_cities, airports = CitySelection_Script(general_params, preprocessor, tier_1_2_cities, './Final_Analysis_Scripts/Temporary_Outputs', './Final_Analysis_Scripts/Temporary_Outputs/PlotlyGraphs')
# print(f"Time taken for CitySelection: {(time.time() - t1):.2f}s")
# print(most_growth_cities)
# print([(x, airports[x].airport_info) for x in airports])