from django.urls import path

from . import views

app_name = "GeneralStats"
urlpatterns = [
    path('', views.GeneralStats, name = 'GeneralStats'),
]