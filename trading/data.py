import numpy as np

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

def generate_stock_price(days, initial_price, volatility, seed=42):
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
        inc = rng.normal(loc = 0, scale = volatility)
        new_price_today = stock_prices[day-1] + inc

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