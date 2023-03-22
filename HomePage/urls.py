from django.urls import path

from . import views

app_name = "HomePage"
urlpatterns = [
    path('', views.HomePage, name='HomePage'),
]