# Generated by Django 4.1.7 on 2023-03-18 17:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('RouteDev', '0008_connection_class_two_way_flight'),
    ]

    operations = [
        migrations.RenameField(
            model_name='route_class',
            old_name='DURATION',
            new_name='DURATION_IN',
        ),
        migrations.AddField(
            model_name='route_class',
            name='DURATION_OUT',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='route_class',
            name='FORECASTED_DEMAND_IN',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='route_class',
            name='FORECASTED_DEMAND_OUT',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='route_class',
            name='GROWTH',
            field=models.DecimalField(decimal_places=2, default=0, max_digits=6),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='route_class',
            name='PRESENT_DEMAND_IN',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='route_class',
            name='PRESENT_DEMAND_OUT',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='route_class',
            name='MARKET_SHARE_IN',
            field=models.DecimalField(decimal_places=2, default=-1, max_digits=5),
        ),
        migrations.AlterField(
            model_name='route_class',
            name='MARKET_SHARE_OUT',
            field=models.DecimalField(decimal_places=2, default=-1, max_digits=5),
        ),
    ]
