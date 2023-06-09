# Generated by Django 4.1.7 on 2023-03-25 21:15

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='NEW_AIRPORT_CLASS',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('NAME', models.CharField(max_length=30)),
                ('INFO', models.CharField(default='', max_length=100)),
                ('GROWTH_RATE', models.FloatField()),
                ('FORECASTED_DEMAND', models.IntegerField()),
                ('TOURISM_GROWTH', models.IntegerField()),
                ('ECONOMIC_GROWTH', models.IntegerField()),
                ('EDUCATION_GROWTH', models.IntegerField()),
                ('POPULATION_GROWTH', models.IntegerField()),
                ('LATITUDE', models.FloatField()),
                ('LONGITUDE', models.FloatField()),
            ],
        ),
    ]
