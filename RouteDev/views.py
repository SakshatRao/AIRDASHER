from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader
from .models import GENERAL_PARAMS_CLASS, ROUTE_PARAMS_CLASS, ROUTE_CLASS, OPTION_CLASS, CITY_CLASS
from AnalysisScripts.CostResourceAnalysis import CostResourceAnalysis_Script

def CitySelection(request):
    return HttpResponse("Hello, world. You're at the RouteDev CitySelection index.")

def RouteSelection(request):
    return HttpResponse("Hello, world. You're at the RouteDev RouteSelection index.")

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
    options_list = CostResourceAnalysis_Script(general_params_dict)
    for option_dict in options_list:
        option = OPTION_CLASS(**option_dict)
        option.ROUTE = route
        print(option)
    print("Saved all options")

    assert(len(GENERAL_PARAMS_CLASS.objects.all()) == 1)
    context = {
        'general_param_info': general_params
    }
    return render(request, 'RouteDev/CostResourceAnalysis.html', context)