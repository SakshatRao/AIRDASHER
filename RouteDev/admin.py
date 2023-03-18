from django.contrib import admin

from .models import GENERAL_PARAMS_CLASS, ROUTE_PARAMS_CLASS, CITY_CLASS, ROUTE_CLASS, OPTION_CLASS, AIRPORT_CLASS, CONNECTION_CLASS

admin.site.register(GENERAL_PARAMS_CLASS)
admin.site.register(ROUTE_PARAMS_CLASS)
admin.site.register(CITY_CLASS)
admin.site.register(ROUTE_CLASS)
admin.site.register(OPTION_CLASS)
admin.site.register(AIRPORT_CLASS)
admin.site.register(CONNECTION_CLASS)