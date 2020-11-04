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
    rev_sample = rev_stock_price[:n]
    n_eff = rev_sample.size
    
    # Storage
    ma = np.zeros(shape=(n_eff, ))

    if n > n_eff:
        print(f'Warning: The first {n_eff} elements of the ma array returned are a ' + \
                f'{n_eff}-day moving average and not a {n}-day moving average')

    # Unweighted moving average
    if len(weights) == 0:
        for i in range(n_eff):
            ma[i] = np.mean(rev_stock_price[i:i+n])
        return ma[::-1]
    else:
        # Ensure valid input
        assert len(weights) == n, "Please provide a weight for every element in moving average calculation"

        # Nomalize and reverse weights 
        weights = np.array(weights)[::-1] / np.sum(weights)
        # Calculate moving average
        for i in range(n_eff):
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
    # Reversing arrays for ease of iteration
    rev_stock_price = stock_price[::-1]
    rev_sample = rev_stock_price[:n]
    n_eff = rev_sample.size

    # Storage
    osc = np.zeros(shape=(n_eff, ))

    if osc_type == 'stochastic':
        for i in range(n_eff):
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
    
    elif osc_type == 'RSI':
        price_diffs_arr = np.zeros(shape = (n_eff, n))
        # Get price differences over past n days
        for i in range(n_eff):
            # Set osc to 0 for when we don't have enough data to calculate RSI
            if rev_stock_price[i:i+n].size != rev_stock_price[i+1:i+1+n].size:
                osc[i] = 0
            else:
                # Get n-day price differences
                price_diffs_arr[i, :] = rev_stock_price[i:i+n] - rev_stock_price[i+1:i+1+n]
                # Find +ve and -ve price diffs and return RSI
                plus_idxs = np.where(price_diffs_arr[i, :] > 0)[0] # Do not include 0 in RSI calculation
                neg_idxs = np.where(price_diffs_arr[i, :] < 0)[0]
                plus_avg = np.mean(price_diffs_arr[i, :][plus_idxs])
                neg_avg = np.abs(np.mean(price_diffs_arr[i, :][neg_idxs]))
                # Handle edge case where RS blows up so RSI tends to 1.0!
                if neg_avg == 0:
                    osc[i] = 1.0
                else:
                    RS = plus_avg / neg_avg
                    RSI = -1/(1 + RS) + 1
                    osc[i] = RSI
        return osc[::-1]








        

        
