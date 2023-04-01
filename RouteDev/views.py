from django.shortcuts import render, redirect
from .models import GENERAL_PARAMS_CLASS, ROUTE_PARAMS_CLASS, ROUTE_CLASS, OPTION_CLASS, CITY_CLASS, AIRPORT_CLASS, CONNECTION_CLASS

import ast
from pathlib import Path
import shutil
import os

from AnalysisScripts.CitySelection import CitySelection_Script, Airport
from AnalysisScripts.RouteSelection import RouteSelection_Script
from AnalysisScripts.CostResourceAnalysis import CostResourceAnalysis_Script
from AnalysisScripts.PreProcessor import PreProcessor

# Load all tier-I/II cities
tier_1_2_cities = [
    'Ahmedabad', 'Bengaluru', 'Mumbai', 'Pune', 'Chennai', 'Hyderabad', 'Kolkata', 'Delhi', 'Visakhapatnam', 'Guwahati', 'Patna',
    'Raipur', 'Gurugram', 'Shimla', 'Jamshedpur', 'Thiruvananthapuram', 'Bhopal', 'Bhubaneswar', 'Amritsar', 'Jaipur', 'Lucknow', 'Dehradun'
]
tier_1_2_cities = tier_1_2_cities + (
    "Guntur, Kakinada, Kurnool, Nellore, Rajamahendravaram, Vijayawada".split(', ')
) + (
    "Bilaspur, Bhilai".split(', ')
) + (
    "Anand, Bhavnagar, Dahod, Jamnagar, Rajkot, Surat, Vadodara".split(', ')
) + (
    "Faridabad, Karnal".split(', ')
) + (
    "Hamirpur".split(', ')
) + (
    "Bokaro Steel City, Dhanbad, Ranchi".split(', ')
) + (
    "Belagavi, Hubballi-Dharwad, Kalaburagi, Mangaluru, Mysuru, Vijayapura".split(', ')
) + (
    "Kannur, Kochi, Kollam, Kozhikode, Malappuram, Thrissur".split(', ')
) + (
    "Gwalior, Indore, Jabalpur, Ratlam, Ujjain".split(', ')
) + (
    "Amravati, Aurangabad, Bhiwandi, Dombivli, Jalgaon, Kolhapur, Nagpur, Nanded, Nashik, Sangli, Solapur, Vasai-Virar".split(', ')
) + (
    "Cuttack, Rourkela".split(', ')
) + (
    "Jalandhar, Ludhiana".split(', ')
) + (
    "Ajmer, Bikaner, Jodhpur".split(', ')
) + (
    "Coimbatore, Erode, Madurai, Salem, Thanjavur, Tiruchirappalli, Tirunelveli, Tiruvannamalai, Vellore".split(', ')
) + (
    "Warangal".split(', ')
) + (
    "Agra, Aligarh, Bareilly, Ghaziabad, Gorakhpur, Jhansi, Kanpur, Mathura, Meerut, Moradabad, Noida, Prayagraj, Varanasi".split(', ')
) + (
    "Asansol, Berhampore, Burdwan, Durgapur, Purulia, Siliguri".split(', ')
) + (
    "Chandigarh, Jammu, Puducherry, Srinagar".split(', ')
)

# If name is too long, we shorten it to not disturb web UI
def shorten_name(name):
    if(len(name) > 12):
        return ''.join(name[:10]) + '...'
    else:
        return name

# Route Development Home Page
def RouteDevHome(request):
    return render(request, 'RouteDev/RouteDevHome.html')

# City Selection Page
def CitySelection(request):

    nothing_changed = True
    first_time = False
    sample_unchanged = True
    forecast_year_unchanged = True
    
    if(request.method == 'POST'):    # If post data sent to view, that means parameter update was made or sample was selected

        NEW_FORECAST_YEAR = int(request.POST['NEW_FORECAST_YEAR'])
        NEW_SAMPLE_NAME = request.POST['NEW_SAMPLE_NAME']
        NEW_ONLY_HUBS = request.POST['NEW_ONLY_HUBS']

        # Update ONLY_HUBS param in GENERAL_PARAMS
        general_params = GENERAL_PARAMS_CLASS.objects.all()[0]
        if(NEW_ONLY_HUBS == 'True'):
            general_params.ONLY_HUBS = True
        else:
            general_params.ONLY_HUBS = False
        general_params.save()
        
        # Check what has changed
        forecast_year_unchanged = (NEW_FORECAST_YEAR == general_params.FORECAST_YEAR)
        sample_unchanged = (NEW_SAMPLE_NAME == general_params.SAMPLE_NAME)
        nothing_changed = forecast_year_unchanged & sample_unchanged

        if(nothing_changed == False):    # If something has changed (either parameter or sample or both)
            
            # Update GENERAL_PARAMS
            general_params = GENERAL_PARAMS_CLASS.objects.all()[0]
            general_params.FORECAST_YEAR = NEW_FORECAST_YEAR
            general_params.SAMPLE_NAME = NEW_SAMPLE_NAME
            general_params.save()
    
    else:    # Else coming to the CitySelection page for the first time
        first_time = True

    if((nothing_changed == False) or (first_time == True)):    # If something has changed (either parameter or sample or both) or if entering page for the first time

        # Run CitySelection_Script
        global tier_1_2_cities
        general_params = GENERAL_PARAMS_CLASS.objects.all()[0]
        THIS_FOLDER = Path(__file__).parent.resolve()
        shutil.rmtree(f'{THIS_FOLDER}/../RouteDev/static/RouteDev/PlotlyGraphs/CitySelection/', ignore_errors = True)
        os.mkdir(f'{THIS_FOLDER}/../RouteDev/static/RouteDev/PlotlyGraphs/CitySelection/')
        preprocessor = PreProcessor(f'{THIS_FOLDER}/../AnalysisScripts/PreProcessed_Datasets')
        cities, airports = CitySelection_Script(
            general_params.__dict__,
            preprocessor, tier_1_2_cities,
            f'{THIS_FOLDER}/../RouteDev/static/RouteDev/ProcessingOutputs',
            f'{THIS_FOLDER}/../RouteDev/static/RouteDev/PlotlyGraphs/CitySelection'
        )

        # Update CITY_CLASS based on cities outputted by CitySelection_Script
        CITY_CLASS.objects.all().delete()
        for city in cities:
            city_params = cities[city]
            city_info = preprocessor.city_info[city]
            city_object = CITY_CLASS(
                NAME = city,
                SHORTENED_NAME = shorten_name(city),
                INFO = city_info,
                AIRPORT_NAME = city_params['Airport'],
                GROWTH_RATE = round(city_params['GrowthRate']),
                FORECASTED_DEMAND = city_params['PredictedFutureTraffic'],
                TOURISM_GROWTH = round(city_params['GrowthRate']),
                ECONOMIC_GROWTH = round(city_params['GrowthRate']),
                EDUCATION_GROWTH = round(city_params['GrowthRate']),
                POPULATION_GROWTH = round(city_params['GrowthRate']),
                LATITUDE = city_params['Latitude'],
                LONGITUDE = city_params['Longitude'],
            )
            city_object.save()
        
        if((sample_unchanged == False) or (len(airports) != len(AIRPORT_CLASS.objects.all()))):    # If sample network got changed

            # Update the AIRPORT_CLASS based on all airports in selected network                
            AIRPORT_CLASS.objects.all().delete()
            for airport in airports:
                airport_params = airports[airport].airport_info
                airport_object = AIRPORT_CLASS(
                    AIRPORT_NAME = airport,
                    NAME = airport_params['City/Town'],
                    SHORTENED_NAME = shorten_name(airport_params['City/Town']),
                    IS_HUB = airport_params['IsHub'],
                    LATITUDE = airport_params['Latitude'],
                    LONGITUDE = airport_params['Longitude'],
                )
                airport_object.save()

            # Update the CONNECTION_CLASS based on all routes in selected network
            CONNECTION_CLASS.objects.all().delete()
            connections_made = []
            all_airport_objects = AIRPORT_CLASS.objects.all()
            for airport1_object in all_airport_objects:
                to_airport_list = airports[airport1_object.AIRPORT_NAME].to_airport_list
                for airport2 in to_airport_list:
                    airport2_object = all_airport_objects.filter(AIRPORT_NAME = airport2.airport_info['Name'])
                    assert(len(airport2_object) == 1)
                    airport2_object = airport2_object[0]
                    connection = CONNECTION_CLASS(NODE1 = airport1_object, NODE2 = airport2_object)
                    if((airport2_object.AIRPORT_NAME, airport1_object.AIRPORT_NAME) not in connections_made):
                        connections_made.append((airport1_object.AIRPORT_NAME, airport2_object.AIRPORT_NAME))
                        connection.TWO_WAY_FLIGHT = False
                    else:
                        connection.TWO_WAY_FLIGHT = True
                    connection.save()

    # Load all required info for HTML display
    general_params = GENERAL_PARAMS_CLASS.objects.all()[0]
    cities = CITY_CLASS.objects.all()
    airports = AIRPORT_CLASS.objects.all()
    connections = CONNECTION_CLASS.objects.filter(TWO_WAY_FLIGHT = False)

    # Load plotly graphs
    divs1 = []
    divs2 = []
    THIS_FOLDER = Path(__file__).parent.resolve()
    for city in cities:
        try:
            div1 = open(f'{THIS_FOLDER}/../RouteDev/static/RouteDev/PlotlyGraphs/CitySelection/{city.NAME}_CitySelection_Graph1.txt', 'r').read()
            div2 = open(f'{THIS_FOLDER}/../RouteDev/static/RouteDev/PlotlyGraphs/CitySelection/{city.NAME}_CitySelection_Graph2.txt', 'r').read()
            divs1.append(div1)
            divs2.append(div2)
        except:
            print("PROBLEM SEEN WITH PLOTLY GRAPHS!")
            pass

    # Send required info to HTML
    context = {
        'general_params_info': general_params,
        'cities_info': cities,
        'cities_div_info': zip(cities, divs1, divs2),
        'airports_info': airports,
        'connections_info': connections
    }
    return render(request, 'RouteDev/CitySelection.html', context)

# Route Selection Page
def RouteSelection(request):

    if(request.method == 'POST'):    # If post data sent to view, that means city has been selected
        
        # Fetch city info which has been selected
        selected_city_name = [x for x in request.POST if x.startswith('SELECTED_CITY_')]
        assert(len(selected_city_name) == 1)
        selected_city_name = selected_city_name[0].split('_')[-1]

        # Fetch info about the previously selected city (if any)
        prev_selected_city = CITY_CLASS.objects.filter(SELECTED = True)
        assert(len(prev_selected_city) <= 1)
        
        # Update selected status of city
        if(len(prev_selected_city) == 1):
            prev_selected_city = prev_selected_city[0]
            prev_selected_city_name = prev_selected_city.NAME
            if(prev_selected_city_name != selected_city_name):
                prev_selected_city.SELECTED = False
                prev_selected_city.save()
        else:
            prev_selected_city_name = ""

        # Fetch city object which has been selected
        new_selected_city = CITY_CLASS.objects.filter(NAME = selected_city_name)[0]
        new_selected_city.SELECTED = True
        new_selected_city.save()

        # Check whether ONLY_HUBS possibly changed
        only_hubs_val_changed = False
        if(len(ROUTE_CLASS.objects.all()) == 0):
            only_hubs_val_changed = True
        general_params = GENERAL_PARAMS_CLASS.objects.all()[0]
        if(general_params.ONLY_HUBS == True):    # If ONLY_HUBS is True now
            if(len(ROUTE_CLASS.objects.filter(AIRPORT__IS_HUB = False)) > 0):    # If ONLY_HUBS was possibly False earlier
                only_hubs_val_changed = True
        else:    # If ONLY_HUBS is False now
            if(len(ROUTE_CLASS.objects.filter(AIRPORT__IS_HUB = False)) == 0):    # If ONLY_HUBS was possibly True earlier
                only_hubs_val_changed = True
        
        if((prev_selected_city_name != selected_city_name) or (only_hubs_val_changed == True)):    # If selected city has changed

            # Fetch info about all airports in network and their connections
            all_airports = AIRPORT_CLASS.objects.all()
            AIRPORT_dict = {}
            for airport_idx, airport in enumerate(all_airports):
                AIRPORT_dict[airport.AIRPORT_NAME] = Airport()
                AIRPORT_dict[airport.AIRPORT_NAME].airport_info = {'Name': airport.AIRPORT_NAME}
            for airport_idx, airport in enumerate(all_airports):
                to_airport_list = CONNECTION_CLASS.objects.filter(NODE1 = airport)
                to_airport_list = [x.NODE2.AIRPORT_NAME for x in to_airport_list]
                from_airport_list = CONNECTION_CLASS.objects.filter(NODE2 = airport)
                from_airport_list = [x.NODE1.AIRPORT_NAME for x in from_airport_list]
                AIRPORT_dict[airport.AIRPORT_NAME].to_airport_list = [AIRPORT_dict[x] for x in to_airport_list]
                AIRPORT_dict[airport.AIRPORT_NAME].from_airport_list = [AIRPORT_dict[x] for x in from_airport_list]

            # Run RouteSelection_Script            
            general_params = GENERAL_PARAMS_CLASS.objects.all()[0]
            global tier_1_2_cities
            THIS_FOLDER = Path(__file__).parent.resolve()
            shutil.rmtree(f'{THIS_FOLDER}/../RouteDev/static/RouteDev/PlotlyGraphs/RouteSelection/', ignore_errors = True)
            os.mkdir(f'{THIS_FOLDER}/../RouteDev/static/RouteDev/PlotlyGraphs/RouteSelection/')
            preprocessor = PreProcessor(f'{THIS_FOLDER}/../AnalysisScripts/PreProcessed_Datasets')
            routes = RouteSelection_Script(
                selected_city_name, AIRPORT_dict,
                general_params.__dict__,
                preprocessor, tier_1_2_cities,
                f'{THIS_FOLDER}/../RouteDev/static/RouteDev/ProcessingOutputs',
                f'{THIS_FOLDER}/../RouteDev/static/RouteDev/PlotlyGraphs/RouteSelection'
            )

            # Save all routes outputted by RouteSelection_Script            
            ROUTE_CLASS.objects.all().delete()
            for route in routes:
                route_params = routes[route]
                route_object = ROUTE_CLASS(
                    CITY = CITY_CLASS.objects.filter(NAME = route_params['City'])[0],
                    AIRPORT = AIRPORT_CLASS.objects.filter(AIRPORT_NAME = route_params['Hub'])[0],
                    DURATION_IN = route_params['IncomingFlightDuration'],
                    DURATION_OUT = route_params['OutgoingFlightDuration'],
                    DURATION = (route_params['IncomingFlightDuration'] + route_params['OutgoingFlightDuration']) // 2,
                    RAILWAYS_NUM = route_params['RailwayNum'],
                    RAILWAYS_DURATION = route_params['RailwayDuration'],
                    RAILWAYS_CAPACITY = route_params['RailwayCapacity'],
                    DISTANCE = route_params['DISTANCE'],
                    PRESENT_DEMAND_IN = route_params['PresentYearInForecast'],
                    PRESENT_DEMAND_OUT = route_params['PresentYearOutForecast'],
                    FORECASTED_DEMAND_IN = route_params['ForecastYearInForecast'],
                    FORECASTED_DEMAND_OUT = route_params['ForecastYearOutForecast'],
                    GROWTH_IN = round(route_params['GrowthIn'], 2),
                    GROWTH_OUT = round(route_params['GrowthOut'], 2),
                    GROWTH = round(route_params['AvgGrowth'], 2),
                    PRICE_IN_MARKET = route_params['PRICE_IN_MARKET'],
                    PRICE_OUT_MARKET = route_params['PRICE_OUT_MARKET'],
                    NUM_IN_MARKET = route_params['NUMBER_PLANES_IN_MARKET'],
                    NUM_OUT_MARKET = route_params['NUMBER_PLANES_OUT_MARKET']
                )
                route_object.save()
    else:    # If post data not available, redirect to CitySelection (since we do not expect to access this page without post data)
        return redirect('RouteDev:CitySelection')

    # Fetch required info for HTML page    
    general_params = GENERAL_PARAMS_CLASS.objects.all()[0]
    selected_city = CITY_CLASS.objects.filter(SELECTED = True)[0]
    routes = ROUTE_CLASS.objects.all()
    airports = []
    for route in routes:
        airports.append(route.AIRPORT)
    def sort_airports_based_on_routes(airports, routes):
        in_routes = [x.AIRPORT for x in routes]
        not_in_routes = [x for x in airports if x not in in_routes]
        return in_routes + not_in_routes
    airports = sort_airports_based_on_routes(airports, routes)
    
    # Fetch plotly graphs
    divs = []
    THIS_FOLDER = Path(__file__).parent.resolve()
    for route in routes:
        try:
            div = open(f'{THIS_FOLDER}/../RouteDev/static/RouteDev/PlotlyGraphs/RouteSelection/{route.CITY.NAME}-{route.AIRPORT.AIRPORT_NAME}_RouteSelection_Graph1.txt', 'r').read()
            divs.append(div)
        except:
            print("PROBLEM SEEN WITH PLOTLY GRAPHS!")
            pass

    # Send required info to HTML
    context = {
        'general_params_info': general_params,
        'selected_city_info': selected_city,
        'routes_info': routes,
        'routes_divs_info': zip(routes, divs),
        'airports_info': airports
    }

    return render(request, 'RouteDev/RouteSelection.html', context)

# Cost & Resource Analysis Page
def CostResourceAnalysis(request):

    # Function to convert market price string to integer (if valid)
    def convert_market_price_to_int(price_str):
        if(price_str == 'N/A'):
            return -1 #Don't care
        else:
            return int(''.join(price_str[1:]))

    # Function to update price & market prices
    #    If a route's market prices are not available, assume price as 100 and range as [5, 200]
    #    If a route's market prices are available, assume price to remain unchanged and range as [market price // 2, market price * 3 // 2]
    def get_route_price_from_market(market_price_in, market_price_out, price_in, price_out):
        if((market_price_in == -1) & (market_price_out == -1)):
            new_price_in_min = 5
            new_price_in = 100
            new_price_in_max = 200
            new_price_out_min = 5
            new_price_out = 100
            new_price_out_max = 200
        else:
            if(market_price_in == -1):
                new_price_in_min = 5
                new_price_in = 100
                new_price_in_max = 200
                new_price_out_min = market_price_out // 2
                new_price_out = price_out
                new_price_out_max = market_price_out * 3 // 2
            elif(market_price_out == -1):
                new_price_in_min = market_price_in // 2
                new_price_in = price_in
                new_price_in_max = market_price_in * 3 // 2
                new_price_out_min = 5
                new_price_out = price_out
                new_price_out_max = 200
            else:
                new_price_in_min = market_price_in // 2
                new_price_in = price_in
                new_price_in_max = market_price_in * 3 // 2
                new_price_out_min = market_price_out // 2
                new_price_out = price_out
                new_price_out_max = market_price_out * 3 // 2
        return new_price_in_min, new_price_in, new_price_in_max, new_price_out_min, new_price_out, new_price_out_max
    
    if(request.method == 'POST'):    # If post data sent to view, that means either route has been selected or parameters have been changed

        params_changed = False
        route_changed = False
        
        if('PARAMS_UPDATE' in request.POST):    # If parameters have been changed

            # Fetch, modify & update the general parameters
            general_params = GENERAL_PARAMS_CLASS.objects.all()[0]
            request_POST = {}
            request_POST['NEW_CAPACITY_NARROWBODY'] = int(request.POST['NEW_CAPACITY_NARROWBODY'])
            request_POST['NEW_CAPACITY_TURBOPROP'] = int(request.POST['NEW_CAPACITY_TURBOPROP'])
            request_POST['NEW_FLEET_NARROWBODY'] = int(request.POST['NEW_FLEET_NARROWBODY'])
            request_POST['NEW_FLEET_TURBOPROP'] = int(request.POST['NEW_FLEET_TURBOPROP'])
            request_POST['NEW_INFLATION_RATE'] = float(request.POST['NEW_INFLATION_RATE'])
            request_POST['NEW_FIXED_COST'] = int(request.POST['NEW_FIXED_COST'])
            request_POST['NEW_OPERATING_COST'] = int(request.POST['NEW_OPERATING_COST'])
            request_POST['NEW_OTHER_COST'] = int(request.POST['NEW_OTHER_COST'])
            request_POST['NEW_MIN_PROFIT_MARGIN'] = float(request.POST['NEW_MIN_PROFIT_MARGIN'])
            request_POST['NEW_ANALYSIS_POINTS'] = str(request.POST['NEW_ANALYSIS_POINTS'])
            request_POST['NEW_DEMAND_FULFILMENT_RATE'] = float(request.POST['NEW_DEMAND_FULFILMENT_RATE'])

            # Check whether general parameters have changed or not
            if(
                (general_params.CAPACITY_NARROWBODY != request_POST['NEW_CAPACITY_NARROWBODY']) or
                (general_params.CAPACITY_TURBOPROP != request_POST['NEW_CAPACITY_TURBOPROP']) or
                (general_params.FLEET_NARROWBODY != request_POST['NEW_FLEET_NARROWBODY']) or
                (general_params.FLEET_TURBOPROP != request_POST['NEW_FLEET_TURBOPROP']) or
                (general_params.INFLATION_RATE != request_POST['NEW_INFLATION_RATE']) or
                (general_params.FIXED_COST != request_POST['NEW_FIXED_COST']) or
                (general_params.OPERATING_COST != request_POST['NEW_OPERATING_COST']) or
                (general_params.OTHER_COST != request_POST['NEW_OTHER_COST']) or
                (general_params.MIN_PROFIT_MARGIN != request_POST['NEW_MIN_PROFIT_MARGIN']) or
                (general_params.ANALYSIS_POINTS != request_POST['NEW_ANALYSIS_POINTS']) or
                (general_params.DEMAND_FULFILMENT_RATE != request_POST['NEW_DEMAND_FULFILMENT_RATE'])
            ):
                params_changed = True

            # Update general parameters            
            general_params.CAPACITY_NARROWBODY = request_POST['NEW_CAPACITY_NARROWBODY']
            general_params.CAPACITY_TURBOPROP = request_POST['NEW_CAPACITY_TURBOPROP']
            general_params.FLEET_NARROWBODY = request_POST['NEW_FLEET_NARROWBODY']
            general_params.FLEET_TURBOPROP = request_POST['NEW_FLEET_TURBOPROP']
            general_params.INFLATION_RATE = request_POST['NEW_INFLATION_RATE']
            general_params.FIXED_COST = request_POST['NEW_FIXED_COST']
            general_params.OPERATING_COST = request_POST['NEW_OPERATING_COST']
            general_params.OTHER_COST = request_POST['NEW_OTHER_COST']
            general_params.MIN_PROFIT_MARGIN = request_POST['NEW_MIN_PROFIT_MARGIN']
            general_params.ANALYSIS_POINTS = request_POST['NEW_ANALYSIS_POINTS']
            general_params.DEMAND_FULFILMENT_RATE = request_POST['NEW_DEMAND_FULFILMENT_RATE']
            general_params.save()

            # Fetch, modify & update the route-specific parameters
            route_params = ROUTE_PARAMS_CLASS.objects.all()[0]
            request_POST['NEW_PRICE_IN'] = int(request.POST['NEW_PRICE_IN'])
            request_POST['NEW_PRICE_OUT'] = int(request.POST['NEW_PRICE_OUT'])

            # Check whether route parameters have changed or not
            if(
                (route_params.PRICE_IN != request_POST['NEW_PRICE_IN']) or
                (route_params.PRICE_OUT != request_POST['NEW_PRICE_OUT'])
            ):
                params_changed = True

            # Update route parameters
            route_params.PRICE_IN = request_POST['NEW_PRICE_IN']
            route_params.PRICE_OUT = request_POST['NEW_PRICE_OUT']
            route_params.save()
        
        else:    # Else assume route has been updated

            # Fetch route which has been selected
            selected_route_name = [x for x in request.POST if x.startswith('SELECTED_ROUTE_')]
            assert(len(selected_route_name) <= 1)

            if(len(selected_route_name) == 1):    # Check if only one route has been selected

                # Fetch route object of selected route
                selected_city_name, selected_hub_name = selected_route_name[0].split('_')[-2:]
                selected_route_name = selected_city_name + '_' + selected_hub_name
                selected_route = ROUTE_CLASS.objects.filter(CITY__NAME = selected_city_name, AIRPORT__AIRPORT_NAME = selected_hub_name)
                assert(len(selected_route) == 1)
                selected_route = selected_route[0]

                # Fetch route object of route which was previously selected (if any)
                prev_selected_route = ROUTE_CLASS.objects.filter(SELECTED = True)
                assert(len(prev_selected_route) <= 1)

                # Update selected status of routes
                if(len(prev_selected_route) == 1):
                    prev_selected_route = prev_selected_route[0]
                    prev_selected_route_name = prev_selected_route.CITY.NAME + '_' + prev_selected_route.AIRPORT.AIRPORT_NAME
                    if((prev_selected_route.CITY.NAME != selected_city_name) | (prev_selected_route.AIRPORT.AIRPORT_NAME != selected_hub_name)):
                        prev_selected_route.SELECTED = False
                        prev_selected_route.save()
                else:
                    prev_selected_route_name = ''
                selected_route.SELECTED = True
                selected_route.save()

            else:    # If more than one route is selected, redirect to City Selection page since we expect only one route to be selected
                return redirect('RouteDev:CitySelection')

            # Check whether selected route has changed
            if(prev_selected_route_name != selected_route_name):
                route_changed = True

        # If parameters or route has changed        
        if(params_changed or route_changed):
            
            # Fetch info about selected route
            selected_route = ROUTE_CLASS.objects.filter(SELECTED = True)[0]
            selected_route_info = {
                'City': selected_route.CITY.NAME,
                'Hub': selected_route.AIRPORT.AIRPORT_NAME,
                'IncomingFlightDuration': selected_route.DURATION_IN,
                'OutgoingFlightDuration': selected_route.DURATION_OUT,
                'PresentYearInForecast': selected_route.PRESENT_DEMAND_IN,
                'PresentYearOutForecast': selected_route.PRESENT_DEMAND_OUT,
                'ForecastYearInForecast': selected_route.FORECASTED_DEMAND_IN,
                'ForecastYearOutForecast': selected_route.FORECASTED_DEMAND_OUT,
                'NUMBER_PLANES_OUT_MARKET': selected_route.NUM_OUT_MARKET,
                'NUMBER_PLANES_IN_MARKET': selected_route.NUM_IN_MARKET,
                'PRICE_OUT_MARKET': convert_market_price_to_int(selected_route.PRICE_OUT_MARKET),
                'PRICE_IN_MARKET': convert_market_price_to_int(selected_route.PRICE_IN_MARKET),
                'DISTANCE': selected_route.DISTANCE,
                'GrowthIn': selected_route.GROWTH_IN,
                'GrowthOut': selected_route.GROWTH_OUT,
                'AvgGrowth': selected_route.GROWTH
            }

            # Fetch info about general parameters
            general_params = GENERAL_PARAMS_CLASS.objects.all()[0]
            general_params_info = general_params.__dict__
            general_params_info['ANALYSIS_POINTS'] = [x-general_params_info['PRESENT_YEAR']-1 for x in ast.literal_eval(general_params_info['ANALYSIS_POINTS'])]

            # Fetch info about route parameters            
            route_params = ROUTE_PARAMS_CLASS.objects.all()[0]
            if(params_changed == False):    # If params has not changed (meaning route has changed), find new prices & price range
                route_params = ROUTE_PARAMS_CLASS.objects.all()[0]
                route_params.PRICE_IN = convert_market_price_to_int(selected_route.PRICE_IN_MARKET)
                route_params.PRICE_OUT = convert_market_price_to_int(selected_route.PRICE_OUT_MARKET)
                _, route_params.PRICE_IN, _, _, route_params.PRICE_OUT, _ = get_route_price_from_market(convert_market_price_to_int(selected_route.PRICE_IN_MARKET), convert_market_price_to_int(selected_route.PRICE_OUT_MARKET), route_params.PRICE_IN, route_params.PRICE_OUT)
                route_params.ROUTE = selected_route
                route_params.save()

            # Run CostResourceAnalysis_Script            
            global tier_1_2_cities
            THIS_FOLDER = Path(__file__).parent.resolve()
            shutil.rmtree(f'{THIS_FOLDER}/../RouteDev/static/RouteDev/PlotlyGraphs/CostResourceAnalysis/', ignore_errors = True)
            os.mkdir(f'{THIS_FOLDER}/../RouteDev/static/RouteDev/PlotlyGraphs/CostResourceAnalysis/')
            preprocessor = PreProcessor(f'{THIS_FOLDER}/../AnalysisScripts/PreProcessed_Datasets')
            options = CostResourceAnalysis_Script(
                selected_route_info,
                general_params.__dict__, route_params.__dict__,
                preprocessor,
                f'{THIS_FOLDER}/../RouteDev/static/RouteDev/ProcessingOutputs',
                f'{THIS_FOLDER}/../RouteDev/static/RouteDev/PlotlyGraphs/CostResourceAnalysis'
            )

            # Save market share info
            selected_route.MARKET_SHARE_IN = options['OtherInfo']['MARKET_SHARE_IN']
            selected_route.MARKET_SHARE_OUT = options['OtherInfo']['MARKET_SHARE_OUT']
            selected_route.save()

            # Save all options
            OPTION_CLASS.objects.all().delete()
            for option_idx, option in enumerate(options['Solutions']):
                option_object = OPTION_CLASS(
                    FEASIBILITY = option['feasibility'],
                    NUM_PLANES = str(option['num_planes']),
                    ROUTE = selected_route,
                    DEMAND = option['cost_resource_analysis']['total_demands'],
                    CAPACITY = round(option['cost_resource_analysis']['total_capacities'], 2),
                    EXPENSES = round(option['cost_resource_analysis']['EXPENSES'], 2),
                    EARNINGS = round(option['cost_resource_analysis']['EARNINGS'], 2),
                    PROFIT_MARGIN = round(option['cost_resource_analysis']['PROFIT_MARGIN_LIST'], 2),
                    PROFITABILITY_YEAR = str(option['cost_resource_analysis']['profitability_year']),
                    OCCUPANCY_RATE = round(option['cost_resource_analysis']['total_flight_vacancies'], 2),
                    RANK = option_idx + 1
                )
                option_object.save()
    
    else:    # If post data not found, redirect to City Selection since we do not expect post data to not be sent
        return redirect('RouteDev:CitySelection')         

    # Fetch required info for HTML display
    general_params = GENERAL_PARAMS_CLASS.objects.all()[0]
    route_params = ROUTE_PARAMS_CLASS.objects.all()[0]
    options = OPTION_CLASS.objects.all()

    # Fetch info about number of planes required and in which sequence should the planes be added to route
    options_total_planes_info = [ast.literal_eval(x.NUM_PLANES)[-1] for x in options]
    options_feasibility = ['feasible' if (x.FEASIBILITY == True) else 'non_feasible' for x in options]
    options_num_planes = [ast.literal_eval(x.NUM_PLANES) for x in options]
    options_plane_addition_info = []
    for option_num_planes in options_num_planes:
        option_plane_addition = [[general_params.PRESENT_YEAR + 1, option_num_planes[0][0], option_num_planes[0][1]]]
        for option_num_planes_idx in range(1, len(option_num_planes)):
            diff_num_narrowbody_planes = option_num_planes[option_num_planes_idx][0] - option_num_planes[option_num_planes_idx - 1][0]
            diff_num_turboprop_planes = option_num_planes[option_num_planes_idx][1] - option_num_planes[option_num_planes_idx - 1][1]
            if((diff_num_narrowbody_planes != 0) | (diff_num_turboprop_planes != 0)):
                option_plane_addition.append([general_params.PRESENT_YEAR + option_num_planes_idx + 1, diff_num_narrowbody_planes, diff_num_turboprop_planes])
        options_plane_addition_info.append(option_plane_addition)

    # Update price range
    selected_route = ROUTE_CLASS.objects.filter(SELECTED = True)[0]
    selected_route_price_in_market = convert_market_price_to_int(selected_route.PRICE_IN_MARKET)
    selected_route_price_out_market = convert_market_price_to_int(selected_route.PRICE_OUT_MARKET)
    selected_route_price_in_min, _, selected_route_price_in_max, selected_route_price_out_min, _, selected_route_price_out_max = get_route_price_from_market(selected_route_price_in_market, selected_route_price_out_market, -1, -1)
    other_param_options = {
        'min_price_in': selected_route_price_in_min,
        'max_price_in': selected_route_price_in_max,
        'min_price_out': selected_route_price_out_min,
        'max_price_out': selected_route_price_out_max
    }

    # Fetch plotly graphs
    divs1 = []
    divs2 = []
    divs3 = []
    THIS_FOLDER = Path(__file__).parent.resolve()
    for option_idx, _ in enumerate(options):
        try:
            div1 = open(f'{THIS_FOLDER}/../RouteDev/static/RouteDev/PlotlyGraphs/CostResourceAnalysis/CostResourceAnalysis_Graph{option_idx + 1}1.txt', 'r').read()
            div2 = open(f'{THIS_FOLDER}/../RouteDev/static/RouteDev/PlotlyGraphs/CostResourceAnalysis/CostResourceAnalysis_Graph{option_idx + 1}2.txt', 'r').read()
            div3 = open(f'{THIS_FOLDER}/../RouteDev/static/RouteDev/PlotlyGraphs/CostResourceAnalysis/CostResourceAnalysis_Graph{option_idx + 1}3.txt', 'r').read()
            divs1.append(div1)
            divs2.append(div2)
            divs3.append(div3)
        except:
            print("PROBLEM SEEN WITH PLOTLY GRAPHS!")
            pass
    
    if(len(options) == 0):
        no_option_returned = True
    else:
        no_option_returned = False
    
    # Send required info to HTML
    context = {
        'general_param_info': general_params,
        'route_param_info': route_params,
        'selected_route_info': selected_route,
        'options_info': zip(options, options_total_planes_info, options_plane_addition_info, options_feasibility, divs1, divs2, divs3),
        'no_option_returned': no_option_returned,
        **other_param_options
    }
    return render(request, 'RouteDev/CostResourceAnalysis.html', context)