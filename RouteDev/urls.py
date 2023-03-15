from django.urls import path

from . import views

app_name = "RouteDev"
urlpatterns = [
    path('CitySelection', views.CitySelection, name='CitySelection'),
    path('RouteSelection', views.RouteSelection, name='RouteSelection'),
    path('CostResourceAnalysis', views.CostResourceAnalysis, name='CostResourceAnalysis'),
]