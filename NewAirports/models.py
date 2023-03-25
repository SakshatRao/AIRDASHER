from django.db import models

class NEW_AIRPORT_CLASS(models.Model):
    NAME = models.CharField(max_length = 30)
    INFO = models.CharField(max_length = 100, default = '')
    FORECASTED_DEMAND = models.IntegerField()
    TOURISM_GROWTH = models.IntegerField()
    ECONOMIC_GROWTH = models.IntegerField()
    EDUCATION_GROWTH = models.IntegerField()
    POPULATION_GROWTH = models.IntegerField()
    LATITUDE = models.FloatField()
    LONGITUDE = models.FloatField()

    def __str__(self):
        return self.NAME