from django.shortcuts import render
from RouteDev.models import AIRPORT_CLASS, CONNECTION_CLASS

import ast
from pathlib import Path
import shutil
import os

from AnalysisScripts.GeneralStats import GeneralStats_Script, Airport
from AnalysisScripts.PreProcessor import PreProcessor

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

def GeneralStats(request):

    global tier_1_2_cities
    THIS_FOLDER = Path(__file__).parent.resolve()
    shutil.rmtree(f'{THIS_FOLDER}/../RouteDev/static/RouteDev/PlotlyGraphs/CitySelection/', ignore_errors = True)
    os.mkdir(f'{THIS_FOLDER}/../RouteDev/static/RouteDev/PlotlyGraphs/CitySelection/')
    preprocessor = PreProcessor(tier_1_2_cities, f'{THIS_FOLDER}/../AnalysisScripts/PreProcessed_Datasets')
    airports = GeneralStats_Script(
        preprocessor, tier_1_2_cities,
        f'{THIS_FOLDER}/../RouteDev/static/RouteDev/ProcessingOutputs',
        f'{THIS_FOLDER}/../RouteDev/static/RouteDev/PlotlyGraphs/CitySelection'
    )

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

    airports = AIRPORT_CLASS.objects.all()
    connections = CONNECTION_CLASS.objects.filter(TWO_WAY_FLIGHT = False)
    THIS_FOLDER = Path(__file__).parent.resolve()
    preprocessor = PreProcessor(tier_1_2_cities, f'{THIS_FOLDER}/../AnalysisScripts/PreProcessed_Datasets')
    div1 = ''
    div2 = ''
    try:
        div1 = open(preprocessor.GeneralStats_div1_path, 'r').read()
        div2 = open(preprocessor.GeneralStats_div2_path, 'r').read()
    except:
        print("PROBLEM SEEN WITH PLOTLY GRAPHS!")
        pass
    
    context = {
        'airports_info': airports,
        'connections_info': connections,
        'div1': div1,
        'div2': div2,
    }
    return render(request, 'GeneralStats/GeneralStats.html', context)