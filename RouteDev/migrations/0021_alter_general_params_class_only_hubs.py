# Generated by Django 4.1.7 on 2023-03-25 14:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('RouteDev', '0020_alter_general_params_class_only_hubs'),
    ]

    operations = [
        migrations.AlterField(
            model_name='general_params_class',
            name='ONLY_HUBS',
            field=models.CharField(default='ONLY_HUBS', max_length=20),
        ),
    ]
