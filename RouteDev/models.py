from django.db import models

class GENERAL_PARAMS_CLASS(models.Model):
    FORECAST_YEAR = models.IntegerField(default = 2033)
    PRESENT_YEAR = models.IntegerField(default = 2023)
    CAPACITY_NARROWBODY = models.IntegerField(default = 300)
    CAPACITY_TURBOPROP = models.IntegerField(default = 100)
    FLEET_NARROWBODY = models.IntegerField(default = 5)
    FLEET_TURBOPROP = models.IntegerField(default = 5)
    INFLATION_RATE = models.FloatField(default = 7.0)
    FIXED_COST = models.IntegerField()
    OPERATING_COST = models.IntegerField()
    OTHER_COST = models.IntegerField()
    MIN_PROFIT_MARGIN = models.FloatField(default = 0)
    ANALYSIS_POINTS = models.CharField(max_length = 200)
    SAMPLE_NAME = models.CharField(max_length = 20)
    ONLY_HUBS = models.BooleanField(default = True)
    DEMAND_FULFILMENT_RATE = models.FloatField(default = 100.0)

class CITY_CLASS(models.Model):
    NAME = models.CharField(max_length = 30)
    INFO = models.CharField(max_length = 100, default = '')
    AIRPORT_NAME = models.CharField(max_length = 3)
    GROWTH_RATE = models.FloatField()
    FORECASTED_DEMAND = models.IntegerField()
    TOURISM_GROWTH = models.IntegerField()
    ECONOMIC_GROWTH = models.IntegerField()
    EDUCATION_GROWTH = models.IntegerField()
    POPULATION_GROWTH = models.IntegerField()
    LATITUDE = models.FloatField()
    LONGITUDE = models.FloatField()
    SELECTED = models.BooleanField(default = False)

    def __str__(self):
        return self.NAME

class AIRPORT_CLASS(models.Model):
    NAME = models.CharField(max_length = 30)
    AIRPORT_NAME = models.CharField(max_length = 3)
    IS_HUB = models.BooleanField(default = False)
    LATITUDE = models.FloatField()
    LONGITUDE = models.FloatField()

    def __str__(self):
        return self.NAME

class CONNECTION_CLASS(models.Model):
    NODE1 = models.ForeignKey(AIRPORT_CLASS, on_delete = models.CASCADE, related_name = 'node1', null = True)
    NODE2 = models.ForeignKey(AIRPORT_CLASS, on_delete = models.CASCADE, related_name = 'node2', null = True)
    TWO_WAY_FLIGHT = models.BooleanField(default = False)

class ROUTE_CLASS(models.Model):
    CITY = models.ForeignKey(CITY_CLASS, on_delete = models.CASCADE, related_name = 'to_city', null = True)
    AIRPORT = models.ForeignKey(AIRPORT_CLASS, on_delete = models.CASCADE, related_name = 'from_city', null = True)
    DURATION_IN = models.IntegerField()
    DURATION_OUT = models.IntegerField()
    DURATION = models.IntegerField(default = 0)
    RAILWAYS_NUM = models.IntegerField(default = 0)
    RAILWAYS_DURATION = models.CharField(default = '', max_length = 10)
    RAILWAYS_CAPACITY = models.CharField(default = '', max_length = 10)
    DISTANCE = models.IntegerField()
    PRESENT_DEMAND_IN = models.IntegerField()
    PRESENT_DEMAND_OUT = models.IntegerField()
    FORECASTED_DEMAND_IN = models.IntegerField()
    FORECASTED_DEMAND_OUT = models.IntegerField()
    GROWTH_IN = models.FloatField()
    GROWTH_OUT = models.FloatField()
    GROWTH = models.FloatField()
    PRICE_IN_MARKET = models.CharField(max_length = 10)
    PRICE_OUT_MARKET = models.CharField(max_length = 10)
    NUM_IN_MARKET = models.IntegerField()
    NUM_OUT_MARKET = models.IntegerField()
    MARKET_SHARE_IN = models.FloatField(default = -1)
    MARKET_SHARE_OUT = models.FloatField(default = -1)
    SELECTED = models.BooleanField(default = False)

    def __str__(self):
        return self.CITY.NAME + "-" + self.AIRPORT.NAME

class ROUTE_PARAMS_CLASS(models.Model):
    PRICE_IN = models.IntegerField()
    PRICE_OUT = models.IntegerField()
    ROUTE = models.ForeignKey(ROUTE_CLASS, on_delete = models.SET_NULL, null = True)

class OPTION_CLASS(models.Model):
    FEASIBILITY = models.BooleanField(default = False)
    NUM_PLANES = models.CharField(max_length = 200)
    ROUTE = models.ForeignKey(ROUTE_CLASS, on_delete = models.CASCADE)
    DEMAND = models.FloatField()
    CAPACITY = models.FloatField()
    EXPENSES = models.FloatField()
    EARNINGS = models.FloatField()
    PROFIT_MARGIN = models.FloatField()
    PROFITABILITY_YEAR = models.CharField(max_length = 5)
    OCCUPANCY_RATE = models.FloatField()
    RANK = models.IntegerField(default = -1)

    def __str__(self):
        return f"{self.ROUTE}-{self.NUM_PLANES}"