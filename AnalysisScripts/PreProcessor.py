import numpy as np
import pandas as pd
import json
import os

##################################################
# PreProcessor
##################################################
# -> Used to store the preprocessed data
##################################################
class PreProcessor:
    
    # input_load_path -> Path of pre-processed datasets
    def __init__(self, input_load_path):

        # Fetching all processed datasets
        print("**************************************")
        print("Loading PreProcessed Datasets")
        self.fetch_PreProcessed_AirRouteDatasets(input_load_path)
        self.fetch_PreProcessed_CityPairWiseDomesticPassengers(input_load_path)
        self.fetch_PreProcessed_IndianRailwaysData(input_load_path)
        self.fetch_PreProcessed_SocioEconomicData(input_load_path)
        self.fetch_PreProcessed_IndiaTourismData(input_load_path)
        self.fetch_Models(input_load_path)
        print("**************************************")

    def fetch_PreProcessed_AirRouteDatasets(self, input_load_path):
        # Datasets Loaded:
        # 1. CityMapping: Mapping cities to respective keys of other datasets
        # 2. IntlCityMapping: Same as above but for APAC dataset
        # 3. FlightConnectionsData_Flights: Data about all Indian flights
        # 4. FlightConnectionsData_IntlFlights: Data about all APAC flights
        # 5. FlightConnectionsData_Airports: Info about Indian airports
        # 6. FlightConnectionsData_IntlAirports: Info about APAC airports
        # 7. SampleAirRouteDatasets: Data about the different sample air networks to find new routes for
        self.city_mapping = pd.read_csv(f'{input_load_path}/CityMapping.csv')
        self.intl_city_mapping = pd.read_csv(f'{input_load_path}/IntlCityMapping.csv')
        self.all_network_data = pd.read_csv(f"{input_load_path}/AirRouteDatasets/FlightConnectionsData_Flights.csv")
        self.all_airport_data = pd.read_csv(f"{input_load_path}/AirRouteDatasets/FlightConnectionsData_Airports.csv")
        self.intl_network_data = pd.read_csv(f"{input_load_path}/AirRouteDatasets/FlightConnectionsData_IntlFlights.csv")
        self.intl_airport_data = pd.read_csv(f"{input_load_path}/AirRouteDatasets/FlightConnectionsData_IntlAirports.csv")
        all_samples_list = os.listdir(f"{input_load_path}/SampleAirRouteDatasets/")
        self.all_samples_network_data = {}
        for sample in all_samples_list:
            network_data = pd.read_csv(f"{input_load_path}/SampleAirRouteDatasets/{sample}")
            network_data.drop('Dummy', axis = 1, inplace = True)
            self.all_samples_network_data[sample.split('.')[0]] = network_data
        print("Loaded AirRouteDatasets")

    def fetch_PreProcessed_CityPairWiseDomesticPassengers(self, input_load_path):
        # Datasets Loaded:
        # 1. CityPairWiseDomesticPassengers: Data about city-pair wise domestic passengers
        self.total_domestic_data = pd.read_csv(f"{input_load_path}/CityPairWiseDomesticPassengers/CityPairWiseDomesticPassengers.csv")
        print("Loaded Domestic Passenger Data")

    def fetch_PreProcessed_IndianRailwaysData(self, input_load_path):
        # Datasets Loaded:
        # 1. all_station_districts: Info about the district each station belongs to
        # 2. CityToCityRoutes: Data about different city-to-city trains
        self.all_station_districts_data = pd.read_csv(f"{input_load_path}/OtherTransportModes/Railways/IndianRailwayStations/all_station_districts.csv")
        with open(f"{input_load_path}/OtherTransportModes/Railways/IndianRailwayRoutes/CityToCityRoutes.json", "r") as load_file:
            self.city_to_city_train_dict = json.load(load_file)
        print("Loaded Indian Railways Data")
    
    def fetch_PreProcessed_SocioEconomicData(self, input_load_path):
        # Datasets Loaded:
        # 1. EconomicData: Data about city's GDP
        # 2. Pop_Area_Household: Data aIndiaPopulation23WithCoordsbout city's population, area, number of households, etc.
        # 3. IndiaPopulation23WithCoords: Data about latest 2023 population of Indian cities
        # 4. EducationData: Data about number of graduates in a city
        # 5. PopulationHistory: Data about historical populations of cities to identify trends
        self.economic_data = pd.read_csv(f"{input_load_path}/IndiaEconomicData/EconomicData.csv")
        self.pop_area_household_data = pd.read_csv(f"{input_load_path}/IndiaSocialData/Pop_Area_Household.csv")
        self.latest_population_data = pd.read_csv(f"{input_load_path}/IndiaSocialData/IndiaPopulation23WithCoords.csv")
        self.education_data = pd.read_csv(f"{input_load_path}/IndiaEducationalData/EducationData.csv")
        with open(f'{input_load_path}/IndiaSocialData/PopulationHistory.json', 'r') as load_file:
            self.population_history_data = json.load(load_file)
        print("Loaded Socio-Economic Data")
    
    def fetch_PreProcessed_IndiaTourismData(self, input_load_path):
        # Datasets Loaded:
        # 1. MonumentVisitors: Data about number of visitors visiting different Indian monuments
        # 2. TouristLocationsCoords: Data about coordinates of different Indian monuments
        self.monument_visitors_data = pd.read_csv(f"{input_load_path}/IndiaTourismData/MonumentVisitors.csv")
        self.tourist_loc_coords_data = pd.read_csv(f"{input_load_path}/IndiaTourismData/TouristLocationsCoords.csv")
        print("Loaded Monument Visitors Data")
    
    def fetch_Models(self, input_load_path):
        # Models loaded:
        # 1. CitySelectionModel_coefs: Linear coefficients of model trained for City Selection
        # 2. CitySelection_cols_standardization: Column standardization weights (i.e. means & stds) used for City Selection training data
        # 3. PCA: PCA coefficients used for City Selection training data
        # 4. data_pca_X: Present PCA-decomposed features used as City Selection training data
        # 5. RouteSelectionModel_coefs: Linear coefficients of model trained for Route Selection
        # 6. city_info: Descriptions for different cities
        # 7. GeneralStats1 & GeneralStats1: Plotly graph divs for General Statistics page
        
        with open(f'{input_load_path}/Models/CitySelectionModel_coefs.json', 'r') as load_file:
            self.CitySelection_model_coefs = json.load(load_file)
        
        with open(f'{input_load_path}/Models/CitySelection_cols_standardization.json', 'r') as load_file:
            self.CitySelection_cols_standardization_vals = json.load(load_file)
        
        all_pca_files = os.listdir(f'{input_load_path}/Models/PCA/')
        
        self.CitySelection_pca = [None] * 10
        for pca_file in all_pca_files:
            pca_idx = int(pca_file.split('.')[0].split('_')[1]) - 1
            self.CitySelection_pca[pca_idx] = np.load(f'{input_load_path}/Models/PCA/{pca_file}')
        
        self.present_features = pd.read_csv(f'{input_load_path}/Models/Present_Features/data_pca_X.csv')

        with open(f'{input_load_path}/Models/RouteSelectionModel_coefs.json', 'r') as load_file:
            self.RouteSelection_model_coefs = json.load(load_file)
        
        with open(f'{input_load_path}/Models/city_info.json', 'r') as load_file:
            self.city_info = json.load(load_file)
        
        self.GeneralStats_div1_path = f'{input_load_path}/Models/GeneralStats1.txt'
        self.GeneralStats_div2_path = f'{input_load_path}/Models/GeneralStats2.txt'