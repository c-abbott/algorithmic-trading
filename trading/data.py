import numpy as np
from os import path

def get_news_drifts(chance, volatility, rng, drift_time): 
    '''
    Simulate the news with %chance.

    Parameters:
        chance, float: 
            Probability of a news event occuring on a particular day.
                
        volatility, float:
            Represents the volatility of a stock.
        
        rng, NumPy random generator object:
            Provides control over the stochastic nature of a news event.
        
        drift_time, tuple:
            Lower and upperbound for samples for news event durations.
    
    Returns:
        drifts, ndarray:
            A variable size numpy array of stock price drifts. Size
            determined int sampled between low and high provided
            by the drift_time variable.
    '''
    news_today = rng.choice([0, 1], p=[1 - chance, chance])

    if news_today:
        # Calculate m and drift
        m = rng.normal(loc=0, scale=2)
        drift = m * volatility

        # Randomly choose the duration
        duration = rng.integers(drift_time[0], drift_time[1])

        # Return drifts due to news event
        drifts = [drift for i in range(duration)]
        return drifts
    else:
        return np.zeros([0]) 

def generate_stock_price(days, initial_price, volatility, seed=None):
    '''
    Generates daily closing share prices for a company,
    for a given number of days.

    Parameters:
        days, int:
            Number of days stock prices are simulated for.
        
        initial_price, float:
            Initial price of stock on day 0.
        
        volatility, float:
            Volatility of a stock.
        
        seed, int (optional):
            Seed for NumPy random number generator.
    
    Returns:
        stock_prices, ndarray:
            Array of size days containing simulated stock prices
            for each day.
    '''
    # Storage arrays
    stock_prices = np.zeros(days)
    total_drift = np.zeros(days + 100) # size overcompensation for robustness

    # Set initial stock prices
    stock_prices[0] = initial_price

    # Define random number generator
    rng = np.random.default_rng(seed = seed)

    # Begin simulating stock prices
    for day in range(1, days):
        # Update price due to volatility of stock
        daily_inc = rng.normal(loc = 0, scale = volatility)
        new_price_today = stock_prices[day-1] + daily_inc

        # Simulate price drifts due to news event
        drifts = get_news_drifts(chance=0.5, volatility=volatility, rng=rng, drift_time=(3, 15))
        duration = len(drifts)
        total_drift[day:day+duration] += drifts

        # Update price due to news event
        new_price_today += total_drift[day]

        # Update overall stock price
        if new_price_today <= 0:
            stock_prices[day] = np.nan
        else:
            stock_prices[day] = new_price_today

    return stock_prices

def get_idxs(items, file_vals):
    col_idxs = []
    for i in range(len(items)):
        diffs = np.abs(file_vals - items[i])
        idx = np.argmin(diffs)
        col_idxs.append(idx)
    return col_idxs

def get_data(method='read', initial_price=None, volatility=None, FILE_NAME='stock_data_5y.txt'):
    '''
    Generates or reads simulation data for one or more stocks over 5 years,
    given their initial share price and volatility.
    
    Input:
        method (str): either 'generate' or 'read' (default 'read').
            If method is 'generate', use generate_stock_price() to generate
                the data from scratch.
            If method is 'read', use Numpy's loadtxt() to read the data
                from the file stock_data_5y.txt.
            
        initial_price (list): list of initial prices for each stock (default None)
            If method is 'generate', use these initial prices to generate the data.
            If method is 'read', choose the column in stock_data_5y.txt with the closest
                starting price to each value in the list, and display an appropriate message.
        
        volatility (list): list of volatilities for each stock (default None).
            If method is 'generate', use these volatilities to generate the data.
            If method is 'read', choose the column in stock_data_5y.txt with the closest
                volatility to each value in the list, and display an appropriate message.
        
        FILE_NAME (string): string holding the relative filename for stock data to simulate (default 'stock_data_5y.txt')
            If method is 'generate', the variable will not be used.
            If method is 'read',

        If no arguments are specified, read price data from the whole file.
        
    Output:
        sim_data (ndarray): NumPy array with N columns, containing the price data
            for the required N stocks each day over 5 years.
    '''
    # Simulate for 5 years discounting leap years
    days = 365 * 5

    if method == 'generate':
        assert initial_price != None, "Please specify the initial price for each stock."
        assert volatility != None, "Please specify the volatility for each stock."
    
        N = len(initial_price)
        # Storage array
        sim_data = np.zeros([days, N])
        for i in range(N):
            sim_data[:, i] = generate_stock_price(days, initial_price[i], volatility[i], seed=42)
        return sim_data

    # Reading data in from file
    else:
        BASE_PATH = path.dirname(__file__)
        FILE_PATH = path.abspath(path.join(BASE_PATH, "..", FILE_NAME))
        data_array = np.loadtxt(FILE_PATH)
        file_vols = data_array[0, :]
        file_ips = data_array[1, :]
        
        if initial_price == None and volatility == None:
            return data_array

        elif initial_price != None and volatility == None:
            N = len(initial_price)
            sim_data = np.zeros([days, N])
            col_idxs = get_idxs(initial_price, file_ips)
            sim_data = data_array[:, col_idxs]
            new_ips = list(sim_data[1, :])
            new_vols = list(data_array[0, col_idxs])
            
            print(f'Found data with initial prices {new_ips} and volatilities {new_vols}.')
            return sim_data
        
        elif initial_price == None and volatility != None:
            N = len(volatility)
            sim_data = np.zeros([days, N])
            col_idxs = get_idxs(volatility, file_vols)
            sim_data = data_array[:, col_idxs]
            
            new_ips = list(sim_data[1, :])
            new_vols = list(data_array[0, col_idxs])
            
            print(f'Found data with initial prices {new_ips} and volatilities {new_vols}.')
            return sim_data

        elif initial_price != None and volatility != None:
            N = len(initial_price)
            sim_data = np.zeros([days, N])
            col_idxs = get_idxs(initial_price, file_ips)
            sim_data = data_array[:, col_idxs]
            new_ips = list(sim_data[1, :])
            new_vols = list(data_array[0, col_idxs])
            print(f'Found data with initial prices {new_ips} and volatilities {new_vols}. ' \
                    'Input argument volatility ignored.')
            return sim_data
    
        
        
if __name__ == "__main__":
    get_data(method='read', initial_price=[210, 58], volatility=[2.4, 2.3])
    get_data(method='read', initial_price=[210, 58])
    get_data(volatility=[5.1])



