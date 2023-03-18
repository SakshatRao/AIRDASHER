# Generated by Django 4.1.7 on 2023-03-18 12:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('RouteDev', '0003_airport_class'),
    ]

    operations = [
        migrations.AddField(
            model_name='airport_class',
            name='LATITUDE',
            field=models.DecimalField(decimal_places=6, default=0, max_digits=10),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='airport_class',
            name='LONGITUDE',
            field=models.DecimalField(decimal_places=6, default=0, max_digits=10),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='city_class',
            name='LATITUDE',
            field=models.DecimalField(decimal_places=6, default=0, max_digits=10),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='city_class',
            name='LONGITUDE',
            field=models.DecimalField(decimal_places=6, default=0, max_digits=10),
            preserve_default=False,
        ),
    ]