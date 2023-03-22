import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.express as px
import numpy as np
import pandas as pd

from pathlib import Path

def CitySelection_GraphingScript(cities):

    demand = np.asarray([10, 20, 20, 30, 50])
    capacity = np.asarray([20, 20, 40, 50, 50])
    years = np.arange(2024, 2029)

    data = [
        go.Scatter(
            x = years, y = demand,
            name = 'Demand', fill = 'tozeroy',
            mode = 'lines', line = dict(color = 'red')
        ),
        go.Scatter(
            x = years, y = capacity,
            name = 'Capacity', fill = 'tozeroy',
            mode = 'lines', line = dict(color = 'blue')
        )
    ]

    layout = go.Layout()

    fig = go.Figure(data = data, layout = layout)
    THIS_FOLDER = Path(__file__).parent.resolve()
    pyo.plot(fig, filename = f"{THIS_FOLDER}/../RouteDev/static/RouteDev/CostDemand.html", auto_open = False, output_type = 'file', include_plotlyjs = 'cdn')

CitySelection_GraphingScript([])