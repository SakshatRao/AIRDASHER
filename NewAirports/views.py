from django.shortcuts import render, redirect
from RouteDev.models import GENERAL_PARAMS_CLASS
from .models import NEW_AIRPORT_CLASS

from pathlib import Path
import shutil
import os

from AnalysisScripts.NewAirport import NewAirport_Script
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

# New Airport Home Page
def NewAirports(request):

    if(request.method == 'POST'):    # If post data sent to view, that means parameter update was made

        NEW_FORECAST_YEAR = int(request.POST['NEW_FORECAST_YEAR'])
        
        # Check whether parameters have changed
        general_params = GENERAL_PARAMS_CLASS.objects.all()[0]
        forecast_year_unchanged = (NEW_FORECAST_YEAR == general_params.FORECAST_YEAR)
        nothing_changed = forecast_year_unchanged
        
        if(nothing_changed == False):    # If something has changed
            
            # Update GENERAL_PARAMS
            general_params.FORECAST_YEAR = NEW_FORECAST_YEAR
            general_params.save()

            # Run NewAirport_Script
            global tier_1_2_cities
            THIS_FOLDER = Path(__file__).parent.resolve()
            shutil.rmtree(f'{THIS_FOLDER}/../NewAirports/static/NewAirports/PlotlyGraphs/CitySelection/', ignore_errors = True)
            os.mkdir(f'{THIS_FOLDER}/../NewAirports/static/NewAirports/PlotlyGraphs/CitySelection/')
            preprocessor = PreProcessor(f'{THIS_FOLDER}/../AnalysisScripts/PreProcessed_Datasets')
            cities = NewAirport_Script(
                general_params.__dict__,
                preprocessor, tier_1_2_cities,
                f'{THIS_FOLDER}/../NewAirports/static/NewAirports/PlotlyGraphs/CitySelection'
            )

            # Save airports
            NEW_AIRPORT_CLASS.objects.all().delete()
            for city in cities:
                city_params = cities[city]
                city_info = preprocessor.city_info[city]
                city_object = NEW_AIRPORT_CLASS(
                    NAME = city,
                    SHORTENED_NAME = shorten_name(city),
                    INFO = city_info,
                    FORECASTED_DEMAND = city_params['PredictedFutureTraffic'],
                    TOURISM_GROWTH = round(city_params['GrowthRate']),
                    ECONOMIC_GROWTH = round(city_params['GrowthRate']),
                    EDUCATION_GROWTH = round(city_params['GrowthRate']),
                    POPULATION_GROWTH = round(city_params['GrowthRate']),
                    LATITUDE = city_params['Latitude'],
                    LONGITUDE = city_params['Longitude'],
                )
                city_object.save()

    # Load all required info for HTML display
    general_params = GENERAL_PARAMS_CLASS.objects.all()[0]
    cities = NEW_AIRPORT_CLASS.objects.all()
    
    # Load plotly graphs
    divs1 = []
    divs2 = []
    THIS_FOLDER = Path(__file__).parent.resolve()
    for city in cities:
        try:
            div1 = open(f'{THIS_FOLDER}/../NewAirports/static/NewAirports/PlotlyGraphs/CitySelection/{city.NAME}_NewAirports_Graph1.txt', 'r').read()
            div2 = open(f'{THIS_FOLDER}/../NewAirports/static/NewAirports/PlotlyGraphs/CitySelection/{city.NAME}_NewAirports_Graph2.txt', 'r').read()
            divs1.append(div1)
            divs2.append(div2)
        except:
            print(city)
            print("PROBLEM SEEN WITH PLOTLY GRAPHS!")
            pass
    
    # Send required info to HTML
    context = {
        'general_params_info': general_params,
        'cities_info': cities,
        'cities_div_info': zip(cities, divs1, divs2),
    }
    return render(request, 'NewAirports/NewAirports.html', context)