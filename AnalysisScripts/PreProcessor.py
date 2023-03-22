import numpy as np
import pandas as pd
import json
import os

class PreProcessor:
    
    def __init__(self, tier_1_2_cities, input_load_path):

        self.tier_1_2_cities = tier_1_2_cities

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
        self.city_mapping = pd.read_csv(f'{input_load_path}/CityMapping.csv')
        self.all_network_data = pd.read_csv(f"{input_load_path}/AirRouteDatasets/FlightConnectionsData_Flights.csv")
        self.all_airport_data = pd.read_csv(f"{input_load_path}/AirRouteDatasets/FlightConnectionsData_Airports.csv")
        all_samples_list = os.listdir(f"{input_load_path}/SampleAirRouteDatasets/")
        self.all_samples_network_data = {}
        for sample in all_samples_list:
            network_data = pd.read_csv(f"{input_load_path}/SampleAirRouteDatasets/{sample}")
            network_data.drop('Dummy', axis = 1, inplace = True)
            self.all_samples_network_data[sample.split('.')[0]] = network_data
        print("Loaded AirRouteDatasets")

    def fetch_PreProcessed_CityPairWiseDomesticPassengers(self, input_load_path):
        self.total_domestic_data = pd.read_csv(f"{input_load_path}/CityPairWiseDomesticPassengers/CityPairWiseDomesticPassengers.csv")
        print("Loaded Domestic Passenger Data")

    def fetch_PreProcessed_IndianRailwaysData(self, input_load_path):
        self.all_station_districts_data = pd.read_csv(f"{input_load_path}/OtherTransportModes/Railways/IndianRailwayStations/all_station_districts.csv")
        with open(f"{input_load_path}/OtherTransportModes/Railways/IndianRailwayRoutes/CityToCityRoutes.json", "r") as load_file:
            self.city_to_city_train_dict = json.load(load_file)
        print("Loaded Indian Railways Data")
    
    def fetch_PreProcessed_SocioEconomicData(self, input_load_path):
        self.economic_data = pd.read_csv(f"{input_load_path}/IndiaEconomicData/EconomicData.csv")
        self.pop_area_household_data = pd.read_csv(f"{input_load_path}/IndiaSocialData/Pop_Area_Household.csv")
        self.latest_population_data = pd.read_csv(f"{input_load_path}/IndiaSocialData/IndiaPopulation23WithCoords.csv")
        self.education_data = pd.read_csv(f"{input_load_path}/IndiaEducationalData/EducationData.csv")
        with open(f'{input_load_path}/IndiaSocialData/PopulationHistory.json', 'r') as load_file:
            self.population_history_data = json.load(load_file)
        print("Loaded Socio-Economic Data")
    
    def fetch_PreProcessed_IndiaTourismData(self, input_load_path):
        self.monument_visitors_data = pd.read_csv(f"{input_load_path}/IndiaTourismData/MonumentVisitors.csv")
        self.tourist_loc_coords_data = pd.read_csv(f"{input_load_path}/IndiaTourismData/TouristLocationsCoords.csv")
        print("Loaded Monument Visitors Data")
    
    def fetch_Models(self, input_load_path):
        # self.CitySelection_model = joblib.load(f"{input_load_path}/Models/CitySelection_model.joblib")
        # self.RouteSelection_model = joblib.load(f"{input_load_path}/Models/RouteSelection_model.joblib")
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