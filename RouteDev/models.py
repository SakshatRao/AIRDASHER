from django.db import models

class GENERAL_PARAMS_CLASS(models.Model):
    FORECAST_YEAR = models.IntegerField(default = 2033)
    PRESENT_YEAR = models.IntegerField(default = 2023)
    CAPACITY_NARROWBODY = models.IntegerField(default = 300)
    CAPACITY_TURBOPROP = models.IntegerField(default = 100)
    FLEET_NARROWBODY = models.IntegerField(default = 5)
    FLEET_TURBOPROP = models.IntegerField(default = 5)
    INFLATION_RATE = models.DecimalField(default = 7.0, max_digits = 5, decimal_places = 2)
    FIXED_COST = models.IntegerField()
    OPERATING_COST = models.IntegerField()
    OTHER_COST = models.IntegerField()
    MIN_PROFIT_MARGIN = models.DecimalField(default = 0, max_digits = 5, decimal_places = 2)
    ANALYSIS_POINTS = models.CharField(max_length = 200)
    SAMPLE_NAME = models.CharField(max_length = 20)

class CITY_CLASS(models.Model):
    NAME = models.CharField(max_length = 30)
    AIRPORT_NAME = models.CharField(max_length = 3)
    GROWTH_RATE = models.DecimalField(max_digits = 6, decimal_places = 2)
    IMG = models.ImageField(blank = True)
    FORECASTED_DEMAND = models.IntegerField()
    TOURISM_GROWTH = models.IntegerField()
    ECONOMIC_GROWTH = models.IntegerField()
    EDUCATION_GROWTH = models.IntegerField()
    POPULATION_GROWTH = models.IntegerField()
    LATITUDE = models.DecimalField(max_digits = 10, decimal_places = 6)
    LONGITUDE = models.DecimalField(max_digits = 10, decimal_places = 6)
    SELECTED = models.BooleanField(default = False)

    def __str__(self):
        return self.NAME

class AIRPORT_CLASS(models.Model):
    NAME = models.CharField(max_length = 30)
    AIRPORT_NAME = models.CharField(max_length = 3)
    IS_HUB = models.BooleanField(default = False)
    LATITUDE = models.DecimalField(max_digits = 10, decimal_places = 6)
    LONGITUDE = models.DecimalField(max_digits = 10, decimal_places = 6)

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
    DISTANCE = models.IntegerField()
    PRESENT_DEMAND_IN = models.IntegerField()
    PRESENT_DEMAND_OUT = models.IntegerField()
    FORECASTED_DEMAND_IN = models.IntegerField()
    FORECASTED_DEMAND_OUT = models.IntegerField()
    GROWTH_IN = models.DecimalField(max_digits = 6, decimal_places = 2)
    GROWTH_OUT = models.DecimalField(max_digits = 6, decimal_places = 2)
    GROWTH = models.DecimalField(max_digits = 6, decimal_places = 2)
    PRICE_IN_MARKET = models.CharField(max_length = 10)
    PRICE_OUT_MARKET = models.CharField(max_length = 10)
    NUM_IN_MARKET = models.IntegerField()
    NUM_OUT_MARKET = models.IntegerField()
    MARKET_SHARE_IN = models.DecimalField(max_digits = 5, decimal_places = 2, default = -1)
    MARKET_SHARE_OUT = models.DecimalField(max_digits = 5, decimal_places = 2, default = -1)
    SELECTED = models.BooleanField(default = False)

    def __str__(self):
        return self.FROM.NAME + "-" + self.TO.NAME

class ROUTE_PARAMS_CLASS(models.Model):
    PRICE_IN = models.IntegerField()
    PRICE_OUT = models.IntegerField()
    ROUTE = models.ForeignKey(ROUTE_CLASS, on_delete = models.SET_NULL, null = True)

class OPTION_CLASS(models.Model):
    FEASIBILITY = models.BooleanField(default = False)
    NUM_PLANES = models.CharField(max_length = 200)
    ROUTE = models.ForeignKey(ROUTE_CLASS, on_delete = models.CASCADE)
    PROFIT_MARGIN = models.DecimalField(max_digits = 6, decimal_places = 2)
    PROFITABILITY_YEAR = models.IntegerField()
    OCCUPANCY_RATE = models.DecimalField(max_digits = 5, decimal_places = 2)
    RANK = models.IntegerField(default = -1)

    def __str__(self):
        return f"{self.ROUTE}-{self.NUM_PLANES}"