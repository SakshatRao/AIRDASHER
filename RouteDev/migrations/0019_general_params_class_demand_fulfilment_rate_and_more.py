# Generated by Django 4.1.7 on 2023-03-25 14:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('RouteDev', '0018_route_class_duration'),
    ]

    operations = [
        migrations.AddField(
            model_name='general_params_class',
            name='DEMAND_FULFILMENT_RATE',
            field=models.FloatField(default=100.0),
        ),
        migrations.AddField(
            model_name='general_params_class',
            name='ONLY_HUBS',
            field=models.BooleanField(default=True),
        ),
    ]
