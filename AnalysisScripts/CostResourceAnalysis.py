import numpy as np
import pandas as pd

def CostResourceAnalysis_Script(general_params):
    return [
        {
            'FEASIBILITY': True,
            'NUM_PLANES': str([(0, 1), (1, 1)]),
            'PROFIT_MARGIN': 56,
            'PROFITABILITY_YEAR': 2026,
            'OCCUPANCY_RATE': 24,
            'RANK': 1
        },
        {
            'FEASIBILITY': False,
            'NUM_PLANES': [(0, 1), (2, 1)],
            'PROFIT_MARGIN': 26,
            'PROFITABILITY_YEAR': 2028,
            'OCCUPANCY_RATE': 21,
            'RANK': 2
        }
    ]