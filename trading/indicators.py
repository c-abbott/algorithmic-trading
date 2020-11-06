import numpy as np

def moving_average(stock_price, window_size=7, weights=[]):
    '''
    Calculates the n-day (possibly weighted) moving average for a given stock over time.
    Note that the moving average is only computed if the moving average calculation is valid.

    Input:
        stock_price (ndarray): single column with the share prices over time for one stock,
            up to the current day.
        window_size (int, default 7): period of the moving average (in days).
        weights (list, default []): must be of length n if specified. Indicates the weights
            to use for the weighted average. If empty, return a non-weighted average.

    Output:
        ma (ndarray): the n-day (possibly weighted) moving average of the share price over time.
    '''
    if len(weights) == 0:
        # Equal weighting and create window of size n
        weights = np.repeat(1.0, window_size) / window_size
        # 'valid' arg ensures that only an n-day MA is taken
        ma = np.convolve(stock_price, weights, 'valid') 
        return ma
    elif len(weights) != 0:
        # Ensure valid input
        assert len(weights) == window_size, "Please provide a weight for every element in moving average calculation"
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        ma = np.convolve(stock_price, weights, 'valid')
        return ma

def oscillator(stock_price, n=7, osc_type='stochastic'):
    '''
    Calculates the level of the stochastic or RSI oscillator with a period of n days.

    Input:
        stock_price (ndarray): single column with the share prices over time for one stock,
            up to the current day.
        n (int, default 7): period of the moving average (in days).
        osc_type (str, default 'stochastic'): either 'stochastic' or 'RSI' to choose an oscillator.

    Output:
        osc (ndarray): the oscillator level with period $n$ for the stock over time.
    '''
    # Reversing arrays for ease of iteration
    rev_stock_price = stock_price[::-1]
    # Storage
    osc = np.zeros(shape=(rev_stock_price.size, ))

    if osc_type == 'stochastic':
        for i in range(rev_stock_price.size):
            # Find low and high prices
            high_price = np.max(rev_stock_price[i:i+n])
            low_price = np.min(rev_stock_price[i:i+n])
            # Compute ingredients for osc
            delta = rev_stock_price[i:i+n][0] - low_price
            delta_max = high_price - low_price
            # Handling division by 0 errors
            if delta == 0 and delta_max == 0:
                osc[i] = 0
            elif delta != 0 and delta_max == 0:
                osc[i] = 1
            elif delta == 0 and delta_max != 0:
                osc[i] = 0
            elif delta != 0 and delta_max != 0:
                # Return stochastic oscillator
                osc[i] = delta / delta_max
        return osc[::-1]
    
    elif osc_type == 'RSI' or osc_type == 'rsi':
        # Calculating all price changes
        deltas = np.diff(stock_price)
        # Calculating RSI for first n days
        seed = deltas[:n+1]
        plus = seed[seed >= 0].sum() / n
        neg = seed[seed < 0].sum() / n
        osc = np.zeros_like(stock_price)
        if plus == 0.:
            osc[:n] = 0.
        elif plus != 0. and neg == 0:
            osc[:n] = 1.
        else:
            rs = plus / neg
            osc[:n] = 1. - 1 / (1. + rs)
            
        # Calculating RSI for rest of days
        for i in range (n, stock_price.size):
            # Get price change
            delta = deltas[i-1]
            # Get correct value to use for avg calculation
            if delta > 0:
                plusval = delta
                negval = 0
            else:
                plusval = 0
                negval = -delta # Negate to get abs value
            # Updating plus and neg average
            plus = (plus * (n-1) + plusval) / n
            neg = (neg * (n-1) + negval) / n
            if plus == 0.:
                osc[i] = 0.
            elif plus != 0. and neg == 0:
                osc[i] = 1.
            else:
                rs = plus / neg
                osc[i] = 1. - 1 / (1. + rs)
        return osc