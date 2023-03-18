from django.shortcuts import render, redirect
from django.template import loader
from .models import GENERAL_PARAMS_CLASS, ROUTE_PARAMS_CLASS, ROUTE_CLASS, OPTION_CLASS, CITY_CLASS, AIRPORT_CLASS, CONNECTION_CLASS

from AnalysisScripts.CitySelection import CitySelection_Script
from AnalysisScripts.RouteSelection import RouteSelection_Script
from AnalysisScripts.PreProcessor import PreProcessor, Airport

tier_1_2_cities = [
    'Ahmedabad', 'Bengaluru', 'Mumbai', 'Pune', 'Chennai', 'Hyderabad', 'Kolkata', 'Delhi', 'Visakhapatnam', 'Guwahati', 'Patna',
    'Raipur', 'Gurugram', 'Shimla', 'Jamshedpur', 'Thiruvananthapuram', 'Bhopal', 'Bhubaneswar', 'Amritsar', 'Jaipur', 'Lucknow', 'Dehradun'
]

def CitySelection(request):

    if(request.method == 'POST'):

        general_params = GENERAL_PARAMS_CLASS.objects.all()[0]

        NEW_FORECAST_YEAR = int(request.POST['NEW_FORECAST_YEAR'])
        NEW_SAMPLE_NAME = request.POST['NEW_SAMPLE_NAME']
        
        forecast_year_unchanged = (NEW_FORECAST_YEAR == general_params.FORECAST_YEAR)
        sample_unchanged = (NEW_SAMPLE_NAME == general_params.SAMPLE_NAME)
        nothing_changed = forecast_year_unchanged & sample_unchanged
        if(nothing_changed == False):
            
            general_params.FORECAST_YEAR = NEW_FORECAST_YEAR
            general_params.SAMPLE_NAME = NEW_SAMPLE_NAME
            general_params.save()

            global tier_1_2_cities
            preprocessor = PreProcessor(tier_1_2_cities, './AnalysisScripts/PreProcessed_Datasets')
            cities, airports = CitySelection_Script(
                general_params.__dict__,
                preprocessor, tier_1_2_cities,
                './RouteDev/static/RouteDev/ProcessingOutputs',
                './RouteDev/static/RouteDev/ProcessingOutputs'
            )

            CITY_CLASS.objects.all().delete()
            for city in cities:
                city_params = cities[city]
                city_object = CITY_CLASS(
                    NAME = city,
                    AIRPORT_NAME = city_params['Airport'],
                    GROWTH_RATE = city_params['GrowthRate'],
                    FORECASTED_DEMAND = city_params['PredictedFutureTraffic'],
                    TOURISM_GROWTH = city_params['GrowthRate'],
                    ECONOMIC_GROWTH = city_params['GrowthRate'],
                    EDUCATION_GROWTH = city_params['GrowthRate'],
                    POPULATION_GROWTH = city_params['GrowthRate'],
                    LATITUDE = city_params['Latitude'],
                    LONGITUDE = city_params['Longitude'],
                )
                city_object.save()
            
            if(sample_unchanged == True):
                
                AIRPORT_CLASS.objects.all().delete()
                for airport in airports:
                    airport_params = airports[airport].airport_info
                    airport_object = AIRPORT_CLASS(
                        AIRPORT_NAME = airport,
                        NAME = airport_params['City/Town'],
                        IS_HUB = airport_params['IsHub'],
                        LATITUDE = airport_params['Latitude'],
                        LONGITUDE = airport_params['Longitude'],
                    )
                    airport_object.save()
                
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

    general_params = GENERAL_PARAMS_CLASS.objects.all()[0]
    cities = CITY_CLASS.objects.all()
    airports = AIRPORT_CLASS.objects.all()
    connections = CONNECTION_CLASS.objects.filter(TWO_WAY_FLIGHT = False)
    context = {
        'general_params_info': general_params,
        'cities_info': cities,
        'airports_info': airports,
        'connections_info': connections
    }
    return render(request, 'RouteDev/CitySelection.html', context)

def RouteSelection(request):

    if(request.method == 'POST'):
        
        selected_city_name = [x for x in request.POST if x.startswith('SELECTED_CITY_')]
        assert(len(selected_city_name) == 1)
        selected_city_name = selected_city_name[0].split('_')[-1]

        prev_selected_city = CITY_CLASS.objects.filter(SELECTED = True)
        assert(len(prev_selected_city) <= 1)
        if(len(prev_selected_city) == 1):
            prev_selected_city = prev_selected_city[0]
            prev_selected_city_name = prev_selected_city.NAME
            if(prev_selected_city_name != selected_city_name):
                prev_selected_city.SELECTED = False
                prev_selected_city.save()
        else:
            prev_selected_city_name = ""
        
        new_selected_city = CITY_CLASS.objects.filter(NAME = selected_city_name)[0]
        new_selected_city.SELECTED = True
        new_selected_city.save()
        
        if(prev_selected_city_name != selected_city_name):

            all_airports = AIRPORT_CLASS.objects.all()
            AIRPORT_dict = {}
            for airport_idx, airport in enumerate(all_airports):
                AIRPORT_dict[airport.AIRPORT_NAME] = Airport(airport_idx)
                AIRPORT_dict[airport.AIRPORT_NAME].airport_info = {'Name': airport.AIRPORT_NAME}
            for airport_idx, airport in enumerate(all_airports):
                to_airport_list = CONNECTION_CLASS.objects.filter(NODE1 = airport)
                to_airport_list = [x.NODE2.AIRPORT_NAME for x in to_airport_list]
                from_airport_list = CONNECTION_CLASS.objects.filter(NODE2 = airport)
                from_airport_list = [x.NODE1.AIRPORT_NAME for x in from_airport_list]
                AIRPORT_dict[airport.AIRPORT_NAME].to_airport_list = [AIRPORT_dict[x] for x in to_airport_list]
                AIRPORT_dict[airport.AIRPORT_NAME].from_airport_list = [AIRPORT_dict[x] for x in from_airport_list]
            
            general_params = GENERAL_PARAMS_CLASS.objects.all()[0]
            global tier_1_2_cities
            preprocessor = PreProcessor(tier_1_2_cities, './AnalysisScripts/PreProcessed_Datasets')
            routes = RouteSelection_Script(
                selected_city_name, AIRPORT_dict,
                general_params.__dict__,
                preprocessor, tier_1_2_cities,
                './AnalysisScripts/PreProcessed_Datasets',
                './AnalysisScripts/PreProcessed_Datasets'
            )
            
            ROUTE_CLASS.objects.all().delete()
            for route in routes:
                route_params = routes[route]
                route_object = ROUTE_CLASS(
                    CITY = CITY_CLASS.objects.filter(NAME = route_params['City'])[0],
                    AIRPORT = AIRPORT_CLASS.objects.filter(AIRPORT_NAME = route_params['Hub'])[0],
                    DURATION_IN = route_params['IncomingFlightDuration'],
                    DURATION_OUT = route_params['OutgoingFlightDuration'],
                    DISTANCE = route_params['DISTANCE'],
                    PRESENT_DEMAND_IN = route_params['PresentYearInForecast'],
                    PRESENT_DEMAND_OUT = route_params['PresentYearOutForecast'],
                    FORECASTED_DEMAND_IN = route_params['ForecastYearInForecast'],
                    FORECASTED_DEMAND_OUT = route_params['ForecastYearOutForecast'],
                    GROWTH_IN = route_params['GrowthIn'],
                    GROWTH_OUT = route_params['GrowthOut'],
                    GROWTH = route_params['AvgGrowth'],
                    PRICE_IN_MARKET = route_params['PRICE_IN_MARKET'],
                    PRICE_OUT_MARKET = route_params['PRICE_OUT_MARKET'],
                    NUM_IN_MARKET = route_params['NUMBER_PLANES_IN_MARKET'],
                    NUM_OUT_MARKET = route_params['NUMBER_PLANES_OUT_MARKET']
                )
                route_object.save()
    else:
        return redirect('RouteDev:CitySelection')
    
    general_params = GENERAL_PARAMS_CLASS.objects.all()[0]
    selected_city = CITY_CLASS.objects.filter(SELECTED = True)[0]
    routes = ROUTE_CLASS.objects.all()
    airports = AIRPORT_CLASS.objects.filter(IS_HUB = True)
    def sort_airports_based_on_routes(airports, routes):
        in_routes = [x.AIRPORT for x in routes]
        not_in_routes = [x for x in airports if x not in in_routes]
        return in_routes + not_in_routes
    airports = sort_airports_based_on_routes(airports, routes)
    context = {
        'general_params_info': general_params,
        'selected_city_info': selected_city,
        'routes_info': routes,
        'airports_info': airports
    }

    return render(request, 'RouteDev/RouteSelection.html', context)

def CostResourceAnalysis(request):

    general_params = GENERAL_PARAMS_CLASS.objects.all()[0]
    route = ROUTE_CLASS.objects.all()[0]

    # For updating parameters
    if(request.method == 'POST'):
        NEW_FORECAST_YEAR = request.POST['NEW_FORECAST_YEAR']
        general_params.FORECAST_YEAR = NEW_FORECAST_YEAR
        general_params.save()
    
    general_params_dict = {
        'FORECAST_YEAR': general_params.FORECAST_YEAR,
        'PRESENT_YEAR': general_params.PRESENT_YEAR
    }
    # options_list = CostResourceAnalysis_AnalysisScript(general_params_dict)
    # for option_dict in options_list:
    #     option = OPTION_CLASS(**option_dict)
    #     option.ROUTE = route
    #     print(option)
    # print("Saved all options")

    assert(len(GENERAL_PARAMS_CLASS.objects.all()) == 1)
    context = {
        'general_param_info': general_params
    }
    return render(request, 'RouteDev/CostResourceAnalysis.html', context)