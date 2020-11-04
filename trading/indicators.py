import numpy as np

def moving_average(stock_price, n=7, weights=[]):
    '''
    Calculates the n-day (possibly weighted) moving average for a given stock over time.

    Input:
        stock_price (ndarray): single column with the share prices over time for one stock,
            up to the current day.
        n (int, default 7): period of the moving average (in days).
        weights (list, default []): must be of length n if specified. Indicates the weights
            to use for the weighted average. If empty, return a non-weighted average.

    Output:
        ma (ndarray): the n-day (possibly weighted) moving average of the share price over time.
    '''
    # Reversing arrays for ease of iteration
    rev_stock_price = stock_price[::-1]
    rev_sample = rev_stock_price[n]
    
    # Storage
    ma = np.zeros(shape=(n, ))

    if n > rev_sample.size:
        print(f'Warning: the first {rev_sample.size} elements of the ma array returned are a' + \
                f'{rev_sample.size}-day moving average and not a {n}-day moving average')

    # Unweighted moving average
    if len(weights) == 0:
        for i in range(rev_sample.size):
            ma[i] = np.mean(rev_stock_price[i:i+n])
        return ma[::-1]
    else:
        # Ensure valid input
        assert len(weights) == n, "Please provide a weight for every element in moving average calculation"

        # Nomalize weights and calculate weighted moving average
        weights = np.array(weights)[::-1] / np.sum(weights)
        for i in range(rev_sample.size):
            ma[i] = np.mean(stock_price[::-1][i:i+n])  * weights[i]
        return ma[::-1]


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
     # Get data from past n days
    sample = stock_price[-n:]

    if sample.size < n:
        print(f'Warning: ')

    if osc_type == 'stochastic':
        # Find low and high prices
        high_price = np.max(sample)
        low_price = np.min(sample)
        # Compute ingredients for osc
        delta = sample[-1] - low_price
        delta_max = high_price - low_price
        # Return stochastic oscillator
        return np.ones(shape=(n, )) * (delta / delta_max)
    
    elif osc_type == 'RSI':
        # Storage
        price_diffs = []

        # Get price differences over past n days
        rev_sample = sample[::-1]
        for i in range(sample.size - 1):
            price_diffs.append(rev_sample[i] - rev_sample[i+1])

        # Find +ve and -ve price diffs and return RSI
        plus_idxs = np.where(price_diffs > 0)[0] # Do not include 0 in RSI calculation
        neg_idxs = np.where(price_diffs < 0)[0]
        plus_avg = np.mean(price_diffs[plus_idxs])
        neg_avg = np.abs(np.mean(price_diffs[neg_idxs]))
        RS = plus_avg / neg_avg
        RSI = -1/(1 + RS) + 1
        return np.ones(shape=(n, )) * RSI






        

        
