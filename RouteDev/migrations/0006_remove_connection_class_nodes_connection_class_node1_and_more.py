# Generated by Django 4.1.7 on 2023-03-18 14:22

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('RouteDev', '0005_connection_class'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='connection_class',
            name='NODES',
        ),
        migrations.AddField(
            model_name='connection_class',
            name='NODE1',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='node1', to='RouteDev.airport_class'),
        ),
        migrations.AddField(
            model_name='connection_class',
            name='NODE2',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='node2', to='RouteDev.airport_class'),
        ),
    ]