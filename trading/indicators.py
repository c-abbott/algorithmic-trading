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
    if len(weights) == 0:
        # Calculating unweighted moving average
        sample = stock_price[-n:]
        ma = np.mean(sample)
        return ma
    else:
        # Ensure valid input
        assert len(weights) == n, "Please provide a weight for every element in moving average calculation"

        # Nomalize weights and weighted moving average
        weights = weights / np.sum(weights)
        weighted_sample = np.multiply(stock_price[-n:], weights)
        ma = np.mean(weighted_sample)
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
    pass
