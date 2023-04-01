import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

##################################################
# CostResourceAnalysis_Script
##################################################
# -> Script for finding ways to introduce
#    new routes and check feasibility and
#    profitability
##################################################
def CostResourceAnalysis_Script(selected_route, general_params, route_params, preprocessor, output_save_path, plotly_save_path):

    # Fetching parameters
    PRESENT_YEAR = general_params['PRESENT_YEAR']
    FORECAST_YEAR = general_params['FORECAST_YEAR']
    INFLATION_RATE = general_params['INFLATION_RATE']
    CAPACITY_NARROWBODY = general_params['CAPACITY_NARROWBODY']
    CAPACITY_TURBOPROP = general_params['CAPACITY_TURBOPROP']
    FIXED_COSTS = general_params['FIXED_COST']
    OPERATING_COSTS = general_params['OPERATING_COST']
    OTHER_COSTS = general_params['OTHER_COST']
    PROFIT_MARGIN = general_params['MIN_PROFIT_MARGIN']
    MARKET_SHARE_PRICE_FACTOR = 4
    FLEET_NARROWBODY = general_params['FLEET_NARROWBODY']
    FLEET_TURBOPROP = general_params['FLEET_TURBOPROP']
    TOP_N_COMBO = 3
    analysis_points = general_params['ANALYSIS_POINTS']
    DEMAND_FULFILMENT_RATE = general_params['DEMAND_FULFILMENT_RATE']

    # Fetching route-specific parameters
    PRICE_IN = route_params['PRICE_IN']
    PRICE_OUT = route_params['PRICE_OUT']

    # Function to perform cost & resource analysis for a given solution (i.e. number of narrowbody & turboprop aircrafts)
    def get_cost_resource_analysis(num_planes, addition_planes, forecasts, PRICE_IN, PRICE_OUT, MARKET_SHARE_IN, MARKET_SHARE_OUT, duration_in, duration_out):
        feasibility = True
        EARNINGS = []

        # Function to include effect of inflation on future costs
        def inflation(price):    # Yearly inflation
            return price * (1 + INFLATION_RATE / 100)
        def inflation_total(price, duration):    # Inflation for a duration of time
            return price * ((1 + INFLATION_RATE / 100) ** duration)

        # Calculate yearly earnings (depends on smaller of demand and capacity)
        current_price_in = PRICE_IN
        current_price_out = PRICE_OUT
        for year_idx, year in enumerate(np.arange(PRESENT_YEAR + 1, FORECAST_YEAR + 1)):
            num_narrow, num_turbo = num_planes[year_idx]
            year_forecasts = forecasts[forecasts['Year'] == year]
            in_demand = year_forecasts.iloc[0]['AdjustedForecastedDemand_InTraffic']
            out_demand = year_forecasts.iloc[0]['AdjustedForecastedDemand_OutTraffic']
            total_capacity_narrow = CAPACITY_NARROWBODY * num_narrow
            total_capacity_turbo = CAPACITY_NARROWBODY * num_turbo
            in_passengers = 0
            out_passengers = 0

            if((total_capacity_narrow + total_capacity_turbo) < in_demand * MARKET_SHARE_IN):    # If capacity is lesser than demand, then earning will be based on how many can fit on the plane
                in_passengers = (total_capacity_narrow + total_capacity_turbo)
            else:    # If demand is lesser than capacity, then earnings will be based on how many people actually travel on plane
                in_passengers = in_demand * MARKET_SHARE_IN
            if((total_capacity_narrow + total_capacity_turbo) < out_demand * MARKET_SHARE_OUT):    # If capacity is lesser than demand, then earning will be based on how many can fit on the plane
                out_passengers = (total_capacity_narrow + total_capacity_turbo)
            else:    # If demand is lesser than capacity, then earnings will be based on how many people actually travel on plane
                out_passengers = out_demand * MARKET_SHARE_OUT
            
            # Add earning of both to & fro flight
            EARNINGS.append((in_passengers * current_price_in) + (out_passengers * current_price_out))

            # Adjust price for inflation for the next year
            current_price_in = inflation(current_price_in)
            current_price_out = inflation(current_price_out)
        total_earnings = np.sum(EARNINGS)
        
        # Operating expense will be -
        #    -> For each day of year
        #    -> For to & fro flights
        #    -> For each turboprop/narrowbody aircraft
        #    -> Depending on hourly operating cost of each type of plane
        #    -> Depending on inflation
        # 
        # NOTE: Operating costs are assumed to be proportional to capacity, hence opex of narrowbody will be opex of turboprop * (capacity of narrowbody / capacity of turboprop)
        operating_expenses_factor = 0
        for idx in range(FORECAST_YEAR - PRESENT_YEAR):
            operating_expenses_factor += ((duration_in + duration_out) / 60) * 365 * (num_planes[idx][0] * (CAPACITY_NARROWBODY / CAPACITY_TURBOPROP) + num_planes[idx][1]) * inflation_total(1, idx)

        # Fixed expense will be -
        #    -> For each purchase of aircrafts on an analysis point
        #    -> Depending on fixed costs of each type of plane
        #    -> Depending on inflation
        # 
        # NOTE: Fixed costs are also assumed to be proportional to capacity, hence fixed costs of narrowbody will be fixed costs of turboprop * (capacity of narrowbody / capacity of turboprop)
        fixed_expenses_factor = 0
        for idx in range(len(addition_planes)):
            fixed_expenses_factor += (addition_planes[idx][0] * CAPACITY_NARROWBODY / CAPACITY_TURBOPROP + addition_planes[idx][1]) * inflation_total(1, FORECAST_YEAR - PRESENT_YEAR - addition_planes[idx][2])
        
        # Other costs are assumed to one-time costs made today
        other_expenses_factor = 1
        
        # Cost equation checks whether expected profit margin is being achieved with current fixed, operating & other costs
        def check_cost_equation(fixed, operating, other):
            profitability = (fixed * fixed_expenses_factor + operating * operating_expenses_factor + other_expenses_factor * other - total_earnings * (1 - PROFIT_MARGIN / 100))
            return (profitability <= 0)
        
        # Calculate total expenses
        EXPENSES = [0] * (FORECAST_YEAR - PRESENT_YEAR)
        EXPENSES[0] += OTHER_COSTS    # Adding other costs to first-year expenses

        # Adding operating expenses
        current_operating_expense = OPERATING_COSTS
        for idx in range(FORECAST_YEAR - PRESENT_YEAR):
            year_operating_expense = current_operating_expense * ((duration_in + duration_out) / 60) * 365 * (num_planes[idx][0] * (CAPACITY_NARROWBODY / CAPACITY_TURBOPROP) + num_planes[idx][1])
            EXPENSES[idx] += year_operating_expense
            current_operating_expense = inflation(current_operating_expense)
        
        # Adding fixed costs
        for idx in range(len(addition_planes)):
            fixed_costs = FIXED_COSTS * (addition_planes[idx][0] * CAPACITY_NARROWBODY / CAPACITY_TURBOPROP + addition_planes[idx][1]) * inflation_total(1, FORECAST_YEAR - PRESENT_YEAR - addition_planes[idx][2])
            EXPENSES[FORECAST_YEAR - PRESENT_YEAR - addition_planes[idx][2]] += fixed_costs
        
        years = np.arange(PRESENT_YEAR + 1, FORECAST_YEAR + 1)
        # Calculating yearly cumulative profit margins
        PROFIT_MARGIN_LIST = (np.cumsum(EARNINGS) - np.cumsum(EXPENSES)) / (np.cumsum(EARNINGS)) * 100

        # Finding yearly total demand, capacity & flight vacancy rates
        total_demands = []
        total_capacities = []
        total_flight_vacancies = []
        for year_idx, year in enumerate(np.arange(PRESENT_YEAR + 1, FORECAST_YEAR + 1)):
            year_forecasts = forecasts[forecasts['Year'] == year]
            total_demand = year_forecasts.iloc[0]['AdjustedForecastedDemand_InTraffic'] * MARKET_SHARE_IN + year_forecasts.iloc[0]['AdjustedForecastedDemand_OutTraffic'] * MARKET_SHARE_OUT
            total_capacity = (num_planes[year_idx][0] * CAPACITY_NARROWBODY + num_planes[year_idx][1] * CAPACITY_TURBOPROP) * 2 * 365
            total_demands.append(total_demand)
            total_capacities.append(total_capacity)
            total_flight_vacancies.append(1 - (total_demand / total_capacity))

        # Check whether given cost parameters satisfy cost equation
        if(check_cost_equation(FIXED_COSTS, OPERATING_COSTS, OTHER_COSTS) == False):    # If cost equation not satisfied, return profitability year as None
            feasibility = False
            return feasibility, {
                'years': years,
                'EXPENSES': EXPENSES,
                'EARNINGS': EARNINGS,
                'PROFIT_MARGIN_LIST': PROFIT_MARGIN_LIST,
                'total_demands': total_demands,
                'total_capacities': total_capacities,
                'total_flight_vacancies': total_flight_vacancies,
                'profitability_year': None
            }
        
        # Else find profitability year
        profitability_year = years[np.where(PROFIT_MARGIN_LIST > 0)[0][0]] - 1        
        return feasibility, {
            'years': years,
            'EXPENSES': EXPENSES,
            'EARNINGS': EARNINGS,
            'PROFIT_MARGIN_LIST': PROFIT_MARGIN_LIST,
            'total_demands': total_demands,
            'total_capacities': total_capacities,
            'total_flight_vacancies': total_flight_vacancies,
            'profitability_year': profitability_year
        }

    def get_route_feasibility(row, PRICE_IN, PRICE_OUT, NUM_IN_MARKET, PRICE_IN_MARKET, NUM_OUT_MARKET, PRICE_OUT_MARKET, SELECTED_CITY, SELECTED_HUB_AIRPORT, selected_city_airport):
        
        # Fetch flight duration
        duration_in = row['IncomingFlightDuration']
        duration_out = row['OutgoingFlightDuration']
        
        # Fetch forecasted demands of given route
        forecast_file = f"{output_save_path}/Forecasted_Route_Demand/City{SELECTED_CITY}_Hub{SELECTED_HUB_AIRPORT}.csv"
        forecasts = pd.read_csv(forecast_file)

        # Step 1: Analyzing Market Share
        # -> Market share is assumed to be dependant on number & price of flights
        # -> Price of flight is available as a parameter, we need to get an idea about roughly how many flights will be required to satisfy demand initially

        # Finding demand that needs to be fulfilled
        # For step 1, it will be first year's demand times demand fulfillment rate
        DEMAND_IN_MAX = forecasts[forecasts['Year'] == PRESENT_YEAR].iloc[0]['AdjustedForecastedDemand_InTraffic'] * DEMAND_FULFILMENT_RATE / 100.0
        DEMAND_OUT_MAX = forecasts[forecasts['Year'] == PRESENT_YEAR].iloc[0]['AdjustedForecastedDemand_OutTraffic'] * DEMAND_FULFILMENT_RATE / 100.0

        # Inequality to be satisfied is -
        # 365 x (n1.C1 + n2.C2) >= Demand x MarketShare(p, n1+n2)
        #    where -
        #    n1 -> number of turboprop flights, n2 -> number of narrowbody flights
        #    C1 -> capacity of turboprop flight, C2 -> capacity of narrowbody flight
        #    Demand -> Total demand of route
        #    MarketShare -> Function to estimate what % of demand will choose our given airline (assumed to depend on price & number of flights for given route)
        #    p -> price of flight
        #
        # NOTE: Market Share is assumed to be of the form -
        #       MarketShare(p, n) = (n/p^4) / ((n/p^4) + (n_market/p_market^4))
        
        # Rearranging terms in the form of -
        #     a.n1^2 + b.n2^2 + 2.h.n1.n2 + 2.g.n1 + 2.f.n2 + c >= 0,
        # We get a, b, h, g, f & c coefficients for to & fro flights
        a_in = 365 * CAPACITY_NARROWBODY * (PRICE_IN_MARKET ** MARKET_SHARE_PRICE_FACTOR)
        b_in = 365 * CAPACITY_TURBOPROP * (PRICE_IN_MARKET ** MARKET_SHARE_PRICE_FACTOR)
        h_in = 365 * (CAPACITY_NARROWBODY * (PRICE_IN_MARKET ** MARKET_SHARE_PRICE_FACTOR) + CAPACITY_TURBOPROP * (PRICE_IN_MARKET ** MARKET_SHARE_PRICE_FACTOR)) / 2
        g_in = ((365 * CAPACITY_NARROWBODY * NUM_IN_MARKET * (PRICE_IN ** MARKET_SHARE_PRICE_FACTOR)) - (DEMAND_IN_MAX * (PRICE_IN_MARKET ** MARKET_SHARE_PRICE_FACTOR))) / 2
        f_in = ((365 * CAPACITY_TURBOPROP * NUM_IN_MARKET * (PRICE_IN ** MARKET_SHARE_PRICE_FACTOR)) - (DEMAND_IN_MAX * (PRICE_IN_MARKET ** MARKET_SHARE_PRICE_FACTOR))) / 2
        c_in = 0

        a_out = 365 * CAPACITY_NARROWBODY * (PRICE_OUT_MARKET ** MARKET_SHARE_PRICE_FACTOR)
        b_out = 365 * CAPACITY_TURBOPROP * (PRICE_OUT_MARKET ** MARKET_SHARE_PRICE_FACTOR)
        h_out = 365 * (CAPACITY_NARROWBODY * (PRICE_OUT_MARKET ** MARKET_SHARE_PRICE_FACTOR) + CAPACITY_TURBOPROP * (PRICE_OUT_MARKET ** MARKET_SHARE_PRICE_FACTOR)) / 2
        g_out = ((365 * CAPACITY_NARROWBODY * NUM_OUT_MARKET * (PRICE_OUT ** MARKET_SHARE_PRICE_FACTOR)) - (DEMAND_OUT_MAX * (PRICE_OUT_MARKET ** MARKET_SHARE_PRICE_FACTOR))) / 2
        f_out = ((365 * CAPACITY_TURBOPROP * NUM_OUT_MARKET * (PRICE_OUT ** MARKET_SHARE_PRICE_FACTOR)) - (DEMAND_OUT_MAX * (PRICE_OUT_MARKET ** MARKET_SHARE_PRICE_FACTOR))) / 2
        c_out = 0

        # Function to find whether given n_narrow & n_turbo satisfy above inequality
        def find_solution(n_narrow, n_turbo, a, b, h, g, f, c):
            return ((a*(n_narrow ** 2)) + (b*(n_turbo ** 2)) + 2*h*n_narrow*n_turbo + 2*g*n_narrow + 2*f*n_turbo + c > 0)

        # This is a special filter which is based on the fact that almost all turboprop flights operate on routes less than 1000km
        # Hence, if given route is >=1000km, we assume turboprop is not viable for this route
        if(selected_route['DISTANCE'] < 621.371):
            turboprop_viability = 1
        else:
            turboprop_viability = 0
        
        solutions_in = []
        for n_narrow in np.arange(0, FLEET_NARROWBODY + 1):    # For possible number of narrowbody flights within fleet
            for n_turbo in np.arange(0, FLEET_TURBOPROP * turboprop_viability + 1):    # For possible number of turboprop flights within fleet (only if route is feasible for turboprop)
                if(find_solution(n_narrow, n_turbo, a_in, b_in, h_in, g_in, f_in, c_in)):    # We find whether given no. of narrowbody & turboprop flights satisfy above inequality, hence deciding whether this solution is feasible initially
                    solutions_in.append((n_narrow, n_turbo))

        solutions_out = []
        for n_narrow in np.arange(0, FLEET_NARROWBODY + 1):    # For possible number of narrowbody flights within fleet
            for n_turbo in np.arange(0, FLEET_TURBOPROP * turboprop_viability + 1):    # For possible number of turboprop flights within fleet (only if route is feasible for turboprop)
                if(find_solution(n_narrow, n_turbo, a_out, b_out, h_out, g_out, f_out, c_out)):    # We find whether given no. of narrowbody & turboprop flights satisfy above inequality, hence deciding whether this solution is feasible initially
                    solutions_out.append((n_narrow, n_turbo))

        # Find intersection of both sets of solutions
        solutions = list(set(solutions_in).intersection(set(solutions_out)))
        
        # Find whether there are any feasible solutions
        feasible = True
        if(len(solutions) == 0):
            feasible = False

        if(feasible == True):    # If feasible solutions exist
            
            # Sort the solutions to find minimum cost of satisfying the inequality
            # -> Cost of operating planes is assumed to be proportional to its capacity, hence sort solutions based on that factor
            sorted_combos = sorted(solutions, key = lambda x: CAPACITY_NARROWBODY * x[0] + CAPACITY_TURBOPROP * x[1])

            # Find the best solution (for initial demand fulfillment)
            # We will use this value to find entry-level market share
            NUM_NARROW, NUM_TURBOPROP = sorted_combos[0]

            assert(a_in * (NUM_NARROW ** 2) + b_in * (NUM_TURBOPROP ** 2) + 2 * h_in * NUM_NARROW * NUM_TURBOPROP + 2 * g_in * NUM_NARROW + 2 * f_in * NUM_TURBOPROP + c_in > 0)
            assert(a_out * (NUM_NARROW ** 2) + b_out * (NUM_TURBOPROP ** 2) + 2 * h_out * NUM_NARROW * NUM_TURBOPROP + 2 * g_out * NUM_NARROW + 2 * f_out * NUM_TURBOPROP + c_out > 0)

            # Find market share
            MARKET_SHARE_IN = ((NUM_NARROW + NUM_TURBOPROP) / (PRICE_IN ** MARKET_SHARE_PRICE_FACTOR)) / (((NUM_NARROW + NUM_TURBOPROP) / (PRICE_IN ** MARKET_SHARE_PRICE_FACTOR)) + (NUM_IN_MARKET / (PRICE_IN_MARKET ** MARKET_SHARE_PRICE_FACTOR)))
            MARKET_SHARE_OUT = ((NUM_NARROW + NUM_TURBOPROP) / (PRICE_OUT ** MARKET_SHARE_PRICE_FACTOR)) / (((NUM_NARROW + NUM_TURBOPROP) / (PRICE_OUT ** MARKET_SHARE_PRICE_FACTOR)) + (NUM_OUT_MARKET / (PRICE_OUT_MARKET ** MARKET_SHARE_PRICE_FACTOR)))
        else:    # If no feasible solutions exist, market share is assumed to be 0
            MARKET_SHARE_IN = 0
            MARKET_SHARE_OUT = 0

        # Step 2: Analyzing overall route feasibility
        # -> Now we have some idea about expected market share
        # -> So to make analysis simple, we assume this market share to remain constant throughout the forecast window

        # Initializing few variables
        DEMAND_IN_MAX_LIST = np.zeros(FORECAST_YEAR - PRESENT_YEAR + 1)
        DEMAND_OUT_MAX_LIST = np.zeros(FORECAST_YEAR - PRESENT_YEAR + 1)
        DEMAND_IN_MIN_LIST = np.zeros(FORECAST_YEAR - PRESENT_YEAR + 1)
        DEMAND_OUT_MIN_LIST = np.zeros(FORECAST_YEAR - PRESENT_YEAR + 1)
        DEMAND_IN_MAX = 0
        DEMAND_OUT_MAX = 0
        DEMAND_IN_MIN = np.inf
        DEMAND_OUT_MIN = np.inf
        count_list = []
        year_idx_list = []
        count = 0

        for year_idx, year in enumerate(np.arange(PRESENT_YEAR + 1, FORECAST_YEAR + 2)):    # Looping through all years + 1 year after FORECAST_YEAR

            if(year_idx in analysis_points):    # If we have to analyze situation in this year, save all important info

                DEMAND_IN_MAX_LIST[year_idx] = DEMAND_IN_MAX
                DEMAND_OUT_MAX_LIST[year_idx] = DEMAND_OUT_MAX
                DEMAND_IN_MIN_LIST[year_idx] = DEMAND_IN_MIN
                DEMAND_OUT_MIN_LIST[year_idx] = DEMAND_OUT_MIN
                count_list.append(count)
                year_idx_list.append(year_idx)

                # Initialize these variables for next window
                count = 0
                DEMAND_IN_MAX = 0
                DEMAND_OUT_MAX = 0
                DEMAND_IN_MIN = np.inf
                DEMAND_OUT_MIN = np.inf

            if(year <= FORECAST_YEAR):    # For all years (except the last iteration)
                
                # Fetch the forecasted demand
                year_forecasts = forecasts[forecasts['Year'] == year]
                in_demand = year_forecasts.iloc[0]['AdjustedForecastedDemand_InTraffic']
                out_demand = year_forecasts.iloc[0]['AdjustedForecastedDemand_OutTraffic']

                # Update max & min demand for given analysis window
                # NOTE: min demand is assumed to be zero, so that demand fulfillment rate makes more intuitive sense that what % of total demand needs to be fulfilled
                if(in_demand > DEMAND_IN_MAX):
                    DEMAND_IN_MAX = in_demand
                if(out_demand > DEMAND_OUT_MAX):
                    DEMAND_OUT_MAX = out_demand
                if(in_demand < DEMAND_IN_MIN):
                    DEMAND_IN_MIN = 0
                if(out_demand < DEMAND_OUT_MIN):
                    DEMAND_OUT_MIN = 0
                count += 1

        # Following functions are to help recursively go through different possible combinations of number of turboprops & narrowbody planes

        # To add b to deepest list in a
        def add_to_deepest_list(a, b):
            if(type(b) == list):
                if(type(b[0]) == list):
                    return [add_to_deepest_list(a, x) for x in b]
                else:
                    assert(type(b[0]) == tuple)
                    return [a] + b

        # Function to help find feasibility of given number of turboprop & narrowbody aircrafts
        def find_solution(n_narrow, n_turbo, a, b, h, g, f, c):
            return ((a*(n_narrow ** 2)) + (b*(n_turbo ** 2)) + 2*h*n_narrow*n_turbo + 2*g*n_narrow + 2*f*n_turbo + c > 0)

        # Function to get all possible solutions for number of turbprop & narrowbody planes over entire forecast window
        # This function will be used recursively to analyze feasiblity of various possible combinations
        #    year_idx -> Which year we are currently at
        #    prev_num_narrowbody -> How many narrowbody planes are currently in operation
        #    prev_num_turbo -> How many turboprop planes are currently in operation
        def get_num_planes(year_idx, prev_num_narrowbody, prev_num_turbo):

            if(year_idx in analysis_points):    # If analysis needs to be done in this year

                # Find what was the max & min demand for the associated analysis window
                DEMAND_IN_MAX = DEMAND_IN_MAX_LIST[year_idx]
                DEMAND_OUT_MAX = DEMAND_OUT_MAX_LIST[year_idx]
                DEMAND_IN_MIN = DEMAND_IN_MIN_LIST[year_idx]
                DEMAND_OUT_MIN = DEMAND_OUT_MIN_LIST[year_idx]

                # Find what demand the airline expects to fulfill
                # This will basically also impact required capacity
                #     So, the DEMAND_FULFILMENT_RATE basically asks what % of max demand do we want to fulfill (whether we want capacity to exceed max demand or whether lesser capacity than 100% demand is also acceptable)
                #     Having lesser demand might reduce total earnings, but it will reduce vacancy rates & operating expenses especially in the initial period of analysis window
                FINAL_DEMAND_IN = (DEMAND_IN_MIN + DEMAND_FULFILMENT_RATE / 100.0 * (DEMAND_IN_MAX - DEMAND_IN_MIN))
                FINAL_DEMAND_OUT = (DEMAND_OUT_MIN + DEMAND_FULFILMENT_RATE / 100.0 * (DEMAND_OUT_MAX - DEMAND_OUT_MIN))

                # Inequality to be satisfied is -
                # 365 x (n1.C1 + n2.C2) >= Demand x MarketShare
                #    where -
                #    n1 -> number of turboprop flights, n2 -> number of narrowbody flights
                #    C1 -> capacity of turboprop flight, C2 -> capacity of narrowbody flight
                #    Demand -> Total demand of route
                #    MarketShare -> What % of demand will choose our given airline (this is already calculated in step 1)
                
                # Rearranging terms in the form of -
                #     a.n1^2 + b.n2^2 + 2.h.n1.n2 + 2.g.n1 + 2.f.n2 + c >= 0,
                # We get a, b, h, g, f & c coefficients for to & fro flights
                a_in = 0
                b_in = 0
                h_in = 0
                g_in = 365 * (CAPACITY_NARROWBODY) / 2
                f_in = 365 * (CAPACITY_TURBOPROP) / 2
                c_in = -MARKET_SHARE_IN * FINAL_DEMAND_IN

                a_out = 0
                b_out = 0
                h_out = 0
                g_out = 365 * (CAPACITY_NARROWBODY) / 2
                f_out = 365 * (CAPACITY_TURBOPROP) / 2
                c_out = -MARKET_SHARE_OUT * FINAL_DEMAND_OUT

                # This is a special filter which is based on the fact that almost all turboprop flights operate on routes less than 1000km
                # Hence, if given route is >=1000km, we assume turboprop is not viable for this route
                if(selected_route['DISTANCE'] < 621.371):
                    turboprop_viability = 1
                else:
                    turboprop_viability = 0
                
                
                solutions_in = []
                for n_narrow in np.arange(prev_num_narrowbody, FLEET_NARROWBODY + 1):    # For possible number of narrowbody flights within fleet
                    for n_turbo in np.arange(prev_num_turbo, FLEET_TURBOPROP * turboprop_viability + 1):    # For possible number of turboprop flights within fleet (only if route is feasible for turboprop)
                        if(find_solution(n_narrow, n_turbo, a_in, b_in, h_in, g_in, f_in, c_in)):    # We find whether given no. of narrowbody & turboprop flights satisfy above inequality, hence deciding whether this solution is feasible initially
                            solutions_in.append((n_narrow, n_turbo))

                solutions_out = []
                for n_narrow in np.arange(prev_num_narrowbody, FLEET_NARROWBODY + 1):    # For possible number of narrowbody flights within fleet
                    for n_turbo in np.arange(prev_num_turbo, FLEET_TURBOPROP * turboprop_viability + 1):    # For possible number of turboprop flights within fleet (only if route is feasible for turboprop)
                        if(find_solution(n_narrow, n_turbo, a_out, b_out, h_out, g_out, f_out, c_out)):    # We find whether given no. of narrowbody & turboprop flights satisfy above inequality, hence deciding whether this solution is feasible initially
                            solutions_out.append((n_narrow, n_turbo))

                # If no feasible solution exists, assign number of narrowbody & turboprop planes as -1 (these values will be dealt with later)
                if(len(solutions_in) == 0):
                    solutions_in = [(-1, -1)]
                    feasible = False

                if(len(solutions_out) == 0):
                    solutions_out = [(-1, -1)]
                    feasible = False

                # Find intersection of both sets of solutions
                solutions = list(set(solutions_in).intersection(set(solutions_out)))

                # If no solution exists, assign number of narrowbody & turboprop planes as -1 (these values will be dealt with later)
                if(len(solutions) == 0):
                    solutions = [(-1, -1)]
                
                # Sort the solutions to find minimum cost of satisfying the inequality
                # -> Cost of operating planes is assumed to be proportional to its capacity, hence sort solutions based on that factor
                # -> Select TOP_N_COMBO (=3) of best solutions
                solutions = sorted(solutions, key = lambda x: CAPACITY_NARROWBODY * x[0] + CAPACITY_TURBOPROP * x[1])[:TOP_N_COMBO]

                all_solutions = []
                for solution in solutions:    # Iterate through all solutions
                    if(year_idx == analysis_points[-1]):    # If this is the last analysis point, then simply append
                        all_solutions.append([solution])
                    else:    # If this is not the last analysis point, then continue analysis with given number of turboprop & narrowbody aircrafts
                        all_solutions.append(add_to_deepest_list((solution[0], solution[1]), get_num_planes(year_idx + 1, solution[0], solution[1])))
                return all_solutions
            else:    # If current year is not analysis point, continue analysis with same inputs
                return get_num_planes(year_idx + 1, prev_num_narrowbody, prev_num_turbo)

        # fetch all possible solutions & flatten it to get single list of solutions
        all_combos = get_num_planes(0, 0, 0)
        def flatten(x, all_combos):
            if(type(x) == list):
                if(type(x[0]) == tuple):
                    all_combos.append(x)
                else:
                    for y in x:
                        all_combos = flatten(y, all_combos)
            return all_combos
        all_combos = flatten(all_combos, [])

        # Ignore those solutions which have -1 in them (indicating they are not feasible)
        all_combos = [x for x in all_combos if np.any([(y[0] == (-1)) | (y[1] == (-1)) for y in x]) == False]

        # Expand these solutions to find number of turboprop & narrowbody aircrafts required for each year and what additions to number of aircrafts are being done at each analysis point
        num_planes_all_combos = []
        addition_planes_all_combos = []
        for combo in all_combos:
            num_planes = [(0, 0)]
            addition_planes = []
            for count_idx, count in enumerate(count_list):
                addition_planes.append((combo[count_idx][0] - num_planes[-1][0], combo[count_idx][1] - num_planes[-1][1], (FORECAST_YEAR - PRESENT_YEAR - year_idx_list[count_idx]) + count_list[count_idx]))
                num_planes.extend([combo[count_idx]] * count)
            num_planes_all_combos.append(num_planes[1:])
            addition_planes_all_combos.append(addition_planes)
        
        # Perform cost analysis on each solution
        combo_info = []
        for combo_idx in range(len(num_planes_all_combos)):
            num_planes = num_planes_all_combos[combo_idx]
            addition_planes = addition_planes_all_combos[combo_idx]
            feasibility, cost_resource_analysis = get_cost_resource_analysis(num_planes, addition_planes, forecasts, PRICE_IN, PRICE_OUT, MARKET_SHARE_IN, MARKET_SHARE_OUT, duration_in, duration_out)    # Find profitability, load factor & other such info for given solution
            combo_info.append({'num_planes': num_planes, 'feasibility': feasibility, 'cost_resource_analysis': cost_resource_analysis})
        
        # Sort all solutions based on final profitability
        # Select top 3 solutions only
        combo_info = list(sorted(combo_info, key = lambda x: x['cost_resource_analysis']['PROFIT_MARGIN_LIST'][-1], reverse = True))[:3]
        
        return {'Solutions': combo_info, 'OtherInfo': {
            'MARKET_SHARE_IN': MARKET_SHARE_IN,
            'MARKET_SHARE_OUT': MARKET_SHARE_OUT
        }}

    # Add one analysis point at end of window
    analysis_points = analysis_points + [FORECAST_YEAR - PRESENT_YEAR]

    city_to_airport_mapping = dict(zip(preprocessor.city_mapping['City'].values, preprocessor.city_mapping['AirRouteData_AirportCode'].values))
        
    # Fetch city & hub info
    SELECTED_CITY = selected_route['City']
    SELECTED_HUB_AIRPORT = selected_route['Hub']
    selected_city_airport = city_to_airport_mapping[SELECTED_CITY]

    # Fetch market info about this route
    route_network_data_out = preprocessor.all_network_data[(preprocessor.all_network_data['From'] == selected_city_airport) & (preprocessor.all_network_data['To'] == SELECTED_HUB_AIRPORT)]
    route_network_data_in = preprocessor.all_network_data[(preprocessor.all_network_data['To'] == selected_city_airport) & (preprocessor.all_network_data['From'] == SELECTED_HUB_AIRPORT)]

    if(route_network_data_out.shape[0] == 0):    # If city->hub route does not exist in market
        NUM_OUT_MARKET = 0
        PRICE_OUT_MARKET = -1 # Don't care
    else:    # Else fetch no. of existing flights & average cheapest price
        PRICE_OUT_MARKET = route_network_data_out['Cheapest Price'].mean()
        NUM_OUT_MARKET = int(route_network_data_out['Number of Flights'].sum())

    if(route_network_data_in.shape[0] == 0):    # If hub->city route does not exist in market
        NUM_IN_MARKET = 0
        PRICE_IN_MARKET = -1 # Don't care
    else:    # Else fetch no. of existing flights & average cheapest price
        PRICE_IN_MARKET = route_network_data_in['Cheapest Price'].mean()
        NUM_IN_MARKET = int(route_network_data_in['Number of Flights'].sum())

    # Get all options for given route
    route_info = get_route_feasibility(selected_route, PRICE_IN, PRICE_OUT, NUM_IN_MARKET, PRICE_IN_MARKET, NUM_OUT_MARKET, PRICE_OUT_MARKET, SELECTED_CITY, SELECTED_HUB_AIRPORT, selected_city_airport)
    route_info['OtherInfo']['PRICE_IN'] = PRICE_IN
    route_info['OtherInfo']['PRICE_OUT'] = PRICE_OUT
    route_info['OtherInfo']['PRICE_IN_MARKET'] = PRICE_IN_MARKET
    route_info['OtherInfo']['PRICE_OUT_MARKET'] = PRICE_OUT_MARKET
    route_info['OtherInfo']['NUM_IN_MARKET'] = NUM_IN_MARKET
    route_info['OtherInfo']['NUM_OUT_MARKET'] = NUM_OUT_MARKET

    trimmed_route_info = route_info.copy()
    for solution_idx in np.arange(len(trimmed_route_info['Solutions'])):    # For each solution

        # Plot the graphs describing different aspects of the solution
        plotly_CostResourceAnalysis(
            solution_idx,
            selected_city_airport, SELECTED_HUB_AIRPORT,
            trimmed_route_info['Solutions'][solution_idx]['cost_resource_analysis']['years'],
            trimmed_route_info['Solutions'][solution_idx]['cost_resource_analysis']['EXPENSES'],
            trimmed_route_info['Solutions'][solution_idx]['cost_resource_analysis']['EARNINGS'],
            trimmed_route_info['Solutions'][solution_idx]['cost_resource_analysis']['PROFIT_MARGIN_LIST'],
            trimmed_route_info['Solutions'][solution_idx]['cost_resource_analysis']['total_demands'],
            trimmed_route_info['Solutions'][solution_idx]['cost_resource_analysis']['total_capacities'],
            trimmed_route_info['Solutions'][solution_idx]['cost_resource_analysis']['total_flight_vacancies'],
            trimmed_route_info['Solutions'][solution_idx]['cost_resource_analysis']['profitability_year'],
            trimmed_route_info['OtherInfo']['MARKET_SHARE_IN'],
            trimmed_route_info['OtherInfo']['MARKET_SHARE_OUT'],
            plotly_save_path
        )

        # Trim the info -> Extracting the final values of all infos
        trimmed_route_info['Solutions'][solution_idx]['cost_resource_analysis']['years'] = trimmed_route_info['Solutions'][solution_idx]['cost_resource_analysis']['years'][-1]
        trimmed_route_info['Solutions'][solution_idx]['cost_resource_analysis']['EXPENSES'] = trimmed_route_info['Solutions'][solution_idx]['cost_resource_analysis']['EXPENSES'][-1]
        trimmed_route_info['Solutions'][solution_idx]['cost_resource_analysis']['EARNINGS'] = trimmed_route_info['Solutions'][solution_idx]['cost_resource_analysis']['EARNINGS'][-1]
        trimmed_route_info['Solutions'][solution_idx]['cost_resource_analysis']['PROFIT_MARGIN_LIST'] = trimmed_route_info['Solutions'][solution_idx]['cost_resource_analysis']['PROFIT_MARGIN_LIST'][-1]
        trimmed_route_info['Solutions'][solution_idx]['cost_resource_analysis']['total_demands'] = trimmed_route_info['Solutions'][solution_idx]['cost_resource_analysis']['total_demands'][-1]
        trimmed_route_info['Solutions'][solution_idx]['cost_resource_analysis']['total_capacities'] = trimmed_route_info['Solutions'][solution_idx]['cost_resource_analysis']['total_capacities'][-1]
        trimmed_route_info['Solutions'][solution_idx]['cost_resource_analysis']['total_flight_vacancies'] = trimmed_route_info['Solutions'][solution_idx]['cost_resource_analysis']['total_flight_vacancies'][-1]

    return trimmed_route_info

# Function to plot graphs for Cost & Resource Analysis stage
def plotly_CostResourceAnalysis(option_idx, CITY_AIRPORT, HUB_AIRPORT, years, EXPENSES, EARNINGS, PROFIT_MARGIN_LIST, total_demands, total_capacities, total_flight_vacancies, profitability_year, MARKET_SHARE_IN, MARKET_SHARE_OUT, plotly_save_path):
    
    # Graph is to visualize market share for to & fro routes
    fig1 = make_subplots(
        rows = 2, cols = 1,
        subplot_titles = [f"FOR {CITY_AIRPORT}→{HUB_AIRPORT} ROUTE", f"FOR {HUB_AIRPORT}→{CITY_AIRPORT} ROUTE"],
        specs=[[{"type": "pie"}], [{"type": "pie"}]]
    )
    
    fig1.add_trace(
        go.Pie(
            labels = ['AIRLINE', 'COMPETITORS'], values = [round(MARKET_SHARE_OUT, 2), 1 - round(MARKET_SHARE_OUT, 2)],
            pull = [0.1, 0],
            marker = dict(colors = ['#2C88D9', '#BBD8F2'], line = dict(color = '#2e353b', width = 1)),
            hoverinfo = "label+percent", textinfo = 'none',
        ),
        row = 1, col = 1
    )
    
    fig1.add_trace(
        go.Pie(
            labels = ['AIRLINE', 'COMPETITORS'], values = [round(MARKET_SHARE_IN, 2), 1 - round(MARKET_SHARE_IN, 2)],
            pull = [0.1, 0],
            marker = dict(colors = ['#2C88D9', '#BBD8F2'], line = dict(color = '#2e353b', width = 1)),
            hoverinfo = "label+percent", textinfo = 'none',
        ),
        row = 2, col = 1
    )
    
    fig1.update_layout(
        title_text = f"<b>MARKET SHARE:</b>",
        height = 700, width = 250,
        paper_bgcolor = '#DBD8FD' , plot_bgcolor = '#DBD8FD',
        titlefont = dict(size = 20),
        margin=dict(l=0,r=0),
        autosize = True,
        showlegend = False
    )
    
    # Graph is to plot & compare yearly cumulative earnings & expenses
    # Also visualizes profitability with time
    if(profitability_year is None):
        text = "PROFIT MARGINS"
    else:
        text = f"PROFITABILITY EXPECTED BY {profitability_year}"
    fig2 = make_subplots(
        rows = 2, cols = 1,
        subplot_titles = [f"CUMULATIVE EARNINGS VS. EXPENSES", text],
    )
    
    fig2.add_trace(
        go.Line(
            x = years, y = np.cumsum(EARNINGS), name = 'Earnings',
            hovertext = [f"Year: {x}<br>Total Cumulative Earnings: ${y:.1f}" for x, y in zip(years, np.cumsum(EARNINGS))],
            hoverinfo = 'text', line = dict(color = '#2C88D9')
        ),
        row = 1, col = 1
    )
    
    fig2.add_trace(
        go.Line(
            x = years, y = np.cumsum(EXPENSES), name = 'Expenses',
            hovertext = [f"Year: {x}<br>Total Cumulative Expenses: ${y:.1f}" for x, y in zip(years, np.cumsum(EXPENSES))],
            hoverinfo = 'text', line = dict(color = '#F7C325')
        ),
        row = 1, col = 1
    )
    
    fig2.update_layout(
        title_text = "<b>PROFITABILITY:</b>",
        height = 700, width = 450,
        paper_bgcolor = '#DBD8FD' , plot_bgcolor = '#DBD8FD',
        titlefont = dict(size = 20),
        shapes=[{
            'type': 'line',
            'x0': years[0],
            'y0': 0,
            'x1': years[-1],
            'y1': 0,
            'yref': 'y2',
            'line': {
                'color': '#2e353b',
                'width': 2,
                'dash': 'dash'
            }
        }],
        spikedistance=1000,
        hoverdistance=100,
        hovermode = 'x',
        legend = dict(
            orientation="h",
            yanchor="top",
            y=0.57,
            xanchor="right",
            x=1,
            itemclick="toggleothers",
            itemdoubleclick="toggle",
        )
    )

    fig2.add_trace(
        go.Line(
            x = years, y = PROFIT_MARGIN_LIST, name = 'Profit Margins',
            hovertext = [f"Year: {x}<br>Profit Margin: {y:.1f}%" for x, y in zip(years, PROFIT_MARGIN_LIST)],
            hoverinfo = 'text', line = dict(color = '#2C88D9'),
            showlegend = False,
            yaxis = 'y2',
            fill = 'tozeroy'
        ),
        row = 2, col = 1
    )
    
    fig2.update_yaxes(
        zeroline = False,
        showgrid = False
    )
    
    fig2.update_xaxes(
        showspikes = True, spikethickness=2, spikecolor="#596673", spikedash='dot', spikemode="across"
    )
    
    # Graph is to show demand vs. capacity trends
    # Also visualizes occupancy rate with time
    fig3 = make_subplots(
        rows = 2, cols = 1,
        subplot_titles = [f"DEMAND VS. CAPACITY", "OCCUPANCY RATE"],
    )
    
    fig3.add_trace(
        go.Line(
            x = years, y = total_capacities, name = 'Capacity',
            hovertext = [f"Year: {x}<br>Total Capacity: {y:.1f}" for x, y in zip(years, total_capacities)],
            hoverinfo = 'text', line = dict(color = '#2C88D9'),
            fill = 'tozeroy',
        ),
        row = 1, col = 1
    )

    fig3.add_trace(
        go.Line(
            x = years, y = total_demands, name = 'Demand',
            hovertext = [f"Year: {x}<br>Total Demand: {y:.1f}" for x, y in zip(years, total_demands)],
            hoverinfo = 'text', line = dict(color = '#F7C325'),
            fill = 'tozeroy'
        ),
        row = 1, col = 1
    )
    
    fig3.add_trace(
        go.Line(
            x = years, y = [np.max([0, np.min([100, 100*(1-x)])]) for x in total_flight_vacancies], name = 'Occupancy Rate',
            hovertext = [f"Year: {x}<br>Occupancy Rate: {y:.1f}%" for x, y in zip(years, [np.max([0, np.min([100, 100*(1-x)])]) for x in total_flight_vacancies])],
            hoverinfo = 'text', line = dict(color = '#2C88D9'),
            showlegend = False
        ),
        row = 2, col = 1
    )
    
    fig3.update_layout(
        title_text = "<b>DEMAND FULFILMENT:</b>",
        height = 700, width = 450,
        paper_bgcolor = '#DBD8FD' , plot_bgcolor = '#DBD8FD',
        titlefont = dict(size = 20),
        spikedistance=1000,
        hoverdistance=100,
        hovermode = 'x',
        legend = dict(
            orientation="h",
            yanchor="top",
            y=0.57,
            xanchor="right",
            x=1,
            itemclick="toggleothers",
            itemdoubleclick="toggle",
        )
    )
    
    fig3.update_yaxes(
        zeroline = False,
        showgrid = False
    )
    
    fig3.update_xaxes(
        showspikes = True, spikethickness=2, spikecolor="#596673", spikedash='dot', spikemode="across"
    )
    
    div1 = pyo.plot(fig1, output_type = 'div', include_plotlyjs = False, show_link = False, link_text = "", config = {"displayModeBar": False, "showTips": False})
    with open(f'{plotly_save_path}/CostResourceAnalysis_Graph{option_idx+1}1.txt', 'w') as save_file:
        save_file.write(div1)
    div2 = pyo.plot(fig2, output_type = 'div', include_plotlyjs = False, show_link = False, link_text = "", config = {"displayModeBar": False, "showTips": False})
    with open(f'{plotly_save_path}/CostResourceAnalysis_Graph{option_idx+1}2.txt', 'w') as save_file:
        save_file.write(div2)
    div3 = pyo.plot(fig3, output_type = 'div', include_plotlyjs = False, show_link = False, link_text = "", config = {"displayModeBar": False, "showTips": False})
    with open(f'{plotly_save_path}/CostResourceAnalysis_Graph{option_idx+1}3.txt', 'w') as save_file:
        save_file.write(div3)