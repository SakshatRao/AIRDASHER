import numpy as np
import pandas as pd

##################################################
# Airport
##################################################
# -> Used for representing an airport in the
#    network (node in graph network)
##################################################
class Airport:
    def __init__(self):
        self.airport_info = {}
        self.to_list = []    # Airport-to-<x> routes
        self.from_list = []    # To-Airport-from-<x> routes
        self.to_airport_list = []    # Destination airports of Airport-to-<x> routes
        self.from_airport_list = []    # Starting airport in To-Airport-from-<x> routes

    def __str__(self):
        return self.airport_info['Name']

    def init_airport_info(self, airport_info):
        self.airport_info = airport_info

    def add_to_list(self, to_route):    # Add to to-list
        self.to_list.append(to_route)
        self.update_to_airports()

    def add_from_list(self, from_route):    # Add to from-list
        self.from_list.append(from_route)
        self.update_from_airports()

    def update_to_airports(self):
        self.to_airport_list = [x.to_airport for x in self.to_list]

    def update_from_airports(self):
        self.from_airport_list = [x.from_airport for x in self.from_list]

##################################################
# Route
##################################################
# -> Used for representing a route in the
#    network (edge in graph network)
##################################################
class Route:
    def __init__(self):
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

##################################################
# GeneralStats_Script
##################################################
# -> Script for returning graph network using
#    all Indian & APAC flights
##################################################
def GeneralStats_Script(preprocessor, tier_1_2_cities_raw):

    intl_city_to_airport_map = dict(zip(preprocessor.intl_city_mapping['City'].values, preprocessor.intl_city_mapping['AirRouteData_AirportCode']))
    city_to_airport_map = dict(zip(preprocessor.city_mapping['City'].values, preprocessor.city_mapping['AirRouteData_AirportCode']))
    all_city_to_airport_map = dict([*intl_city_to_airport_map.items()] + [*city_to_airport_map.items()])
    tier_1_2_cities = [x for x in tier_1_2_cities_raw if pd.isnull(all_city_to_airport_map[x]) == False]
    tier_1_2_cities_airports = preprocessor.city_mapping['AirRouteData_AirportCode'].dropna().values

    # Get individual airport info
    airport_df = pd.concat([preprocessor.all_airport_data, preprocessor.intl_airport_data], axis = 0).reset_index(drop = True)
    airport_df['IsHub'] = pd.Series(np.ones(airport_df.shape[0]))
    airport_df = airport_df[['Name', 'City/Town', 'IsHub']]

    # Function to represent graph network as nodes & edges
    # Input variables:
    #    raw_route_df -> Routes/edges information
    #    raw_airport_df -> Airport/nodes information
    #    USE_ONLY_SELECTIVE_CITIES -> Whether to exclude airports which are not for tier-I/II cities
    #    to_use_airports -> List of tier-I/II cities
    def plot_network(
        raw_route_df, raw_airport_df,
        USE_ONLY_SELECTIVE_CITIES = True, to_use_airports = []
    ):
        route_df = raw_route_df.copy()
        route_attr_cols = [x for x in route_df.columns if x not in ['From', 'To']]

        airport_df = raw_airport_df.copy()

        if(USE_ONLY_SELECTIVE_CITIES):
            # Filter for tier-I/II cities
            exclude_idx = []
            for idx, row in route_df.iterrows():
                if((row['From'] not in to_use_airports) or (row['To'] not in to_use_airports)):
                    exclude_idx.append(idx)
            route_df = route_df.drop(exclude_idx, axis = 0).reset_index(drop = True)

            exclude_idx = []
            for idx, row in airport_df.iterrows():
                if(row['Name'] not in to_use_airports):
                    exclude_idx.append(idx)
            airport_df = airport_df.drop(exclude_idx, axis = 0).reset_index(drop = True)

        # Extract each airport's coordinates
        airport_to_coord_map = dict(zip(preprocessor.city_mapping['AirRouteData_AirportCode'].values, preprocessor.city_mapping['Airport_City_Coords']))
        intl_airport_to_coord_map = dict(zip(preprocessor.intl_city_mapping['AirRouteData_AirportCode'].values, preprocessor.intl_city_mapping['Airport_City_Coords']))
        all_airport_to_coord_map = dict([*airport_to_coord_map.items()] + [*intl_airport_to_coord_map.items()])
        airport_df['Latitude_Longitude'] = airport_df['Name'].map(all_airport_to_coord_map)
        airport_df['Latitude'] = airport_df['Latitude_Longitude'].apply(lambda x: float(x.split(', ')[0].strip()))
        airport_df['Longitude'] = airport_df['Latitude_Longitude'].apply(lambda x: float(x.split(', ')[1].strip()))
        airport_df = airport_df.drop('Latitude_Longitude', axis = 1)
        airport_attr_cols = [x for x in airport_df.columns]

        AIRPORTS = {}
        ROUTES = {}

        # Create nodes for each airport
        for idx, row in airport_df.iterrows():
            airport_obj = Airport()
            airport_attr = dict([(x, row[x]) for x in airport_attr_cols])
            airport_obj.init_airport_info(airport_attr)
            AIRPORTS[row['Name']] = airport_obj

        # Create edges for each route
        for idx, row in route_df.iterrows():
            route_obj = Route()
            AIRPORTS[row['From']], AIRPORTS[row['To']] = route_obj.init_from_to(AIRPORTS[row['From']], AIRPORTS[row['To']])
            route_attr = dict([(x, row[x]) for x in route_attr_cols])
            route_obj.init_route_info(route_attr)
            ROUTES[f"{row['From']}-{row['To']}"] = route_obj

        return AIRPORTS, ROUTES

    AIRPORTS, ROUTES = plot_network(
        pd.concat([preprocessor.all_network_data, preprocessor.intl_network_data], axis = 0).reset_index(drop = True),
        airport_df,
        USE_ONLY_SELECTIVE_CITIES = True, to_use_airports = [*tier_1_2_cities_airports] + [*preprocessor.intl_airport_data['Name'].values]
    )
    
    return AIRPORTS