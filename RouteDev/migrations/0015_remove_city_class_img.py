# Generated by Django 4.1.7 on 2023-03-22 16:50

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('RouteDev', '0014_alter_option_class_profitability_year'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='city_class',
            name='IMG',
        ),
    ]
