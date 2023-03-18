import numpy as np
import pandas as pd
import json
import os
import joblib

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
        self.CitySelection_model = joblib.load(f"{input_load_path}/Models/CitySelection_model.joblib")
        self.RouteSelection_model = joblib.load(f"{input_load_path}/Models/RouteSelection_model.joblib")

class ID_Generator:
    def __init__(self, max_range = 100):
        self.all_ids = []
        self.max_range = max_range
    
    def generate_id(self):
        new_id = 0
        while(True):
            new_id = np.random.randint(self.max_range)
            if(new_id not in self.all_ids):
                break
        self.all_ids.append(new_id)
        return new_id

class Airport:
    def __init__(self, airport_id):
        self.id = airport_id
        self.airport_info = {}
        self.to_list = []
        self.from_list = []
        self.to_airport_list = []
        self.from_airport_list = []
        
    def __str__(self):
        return self.airport_info['Name']
        
    def init_airport_info(self, airport_info):
        self.airport_info = airport_info
    
    def add_to_list(self, to_route):
        self.to_list.append(to_route)
        self.update_to_airports()
        
    def add_from_list(self, from_route):
        self.from_list.append(from_route)
        self.update_from_airports()
        
    def update_to_airports(self):
        self.to_airport_list = [x.to_airport for x in self.to_list]
    
    def update_from_airports(self):
        self.from_airport_list = [x.from_airport for x in self.from_list]

class Route:
    def __init__(self, route_id):
        self.id = route_id
        self.from_airport = None
        self.to_airport = None
        self.route_info = {}
    
    def __str__(self):
        return str(self.from_airport) + "-" + str(self.to_airport)
    
    def init_from_to(self, from_airport, to_airport):
        self.from_airport = from_airport
        self.to_airport = to_airport
        return self.update_from_to_list()
    
    def init_route_info(self, route_info):
        self.route_info = route_info
    
    def update_from_to_list(self):
        self.to_airport.add_from_list(self)
        self.from_airport.add_to_list(self)
        return self.from_airport, self.to_airport