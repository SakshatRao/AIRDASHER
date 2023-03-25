from django.urls import path

from . import views

app_name = "NewAirports"
urlpatterns = [
    path('', views.NewAirports, name = 'NewAirports'),
]