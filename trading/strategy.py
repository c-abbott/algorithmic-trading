# Functions to implement our trading strategy.
import numpy as np
from trading import process as proc
from trading import indicators as indi
from trading import data as data

def sell_all_stock(stock_prices, fees, portfolio, ledger):
    '''
    Sell all stock on the final day.

    Input:
        stock_prices (ndarray): the stock price data
        fees (float, default 20): transaction fees
        portfolio (list): our current portfolio
        ledger (str): path to the ledger file
    
    Output: None
    '''
    # Number of stocks simulated
    N = int(stock_prices.shape[1])
    # Finish with selling all stock on final day
    for stock_id in range(N):
        proc.sell(stock_prices.shape[0]-1, stock_id, stock_prices, fees, portfolio, ledger)

def random(stock_prices, period=7, amount=5000, fees=20, ledger='ledger_random.txt'):
    '''
    Randomly decide, every period, which stocks to purchase,
    do nothing, or sell (with equal probability).
    Spend a maximum of amount on every purchase.

    Input:
        stock_prices (ndarray): the stock price data
        period (int, default 7): how often we buy/sell (days)
        amount (float, default 5000): how much we spend on each purchase
            (must cover fees)
        fees (float, default 20): transaction fees
        ledger (str): path to the ledger file

    Output: None
    '''
    # Number of stock to simulate
    N = int(stock_prices.shape[1])
    # Create day 0 portfolio
    portfolio = proc.create_portfolio(np.repeat(amount, N), stock_prices, fees, ledger)
    # Determine dates on which we act
    action_days = np.arange(1, stock_prices.shape[0]+1, period)

    for action_day in action_days:
        for stock_id in range(N):
            # Randomly sample action from discrete unif dist.
            action = np.random.choice([0, 1, 2])
            if action == 1: # Buying
                proc.buy(action_day, stock_id, amount, stock_prices, fees, portfolio, ledger)
            elif action == 2: # Selling
                proc.sell(action_day, stock_id, stock_prices, fees, portfolio, ledger)

    # Finish with selling all stock on final day
    sell_all_stock(stock_prices, fees, portfolio, ledger)

def crossing_averages(stock_prices, sma_window=200, fma_window=50, amount=5000, fees=20, ledger='ledger_cross.txt'):
    '''
    Algorithmic trading strategy based on the crossing of the slow moving average (SMA) and fast
    moving average (FMA).
        - When the FMA crosses the SMA from below we buy shares.
        - When the FMA crosses the SMA from above we sell shares

    Input:
        stock_prices (ndarray): the stock price data
        sma_period (int, default 200): days used to calculate SMA
        fma_period (int, default 50): days used to calculate FMA
        amount (float, default 5000): how much we spend on each purchase (must cover fees)
        fees (float, default 20): transaction fees
        ledger (str): path to the ledger file

    Output: 
        None
    '''
    # Number of stocks simulated
    N = int(stock_prices.shape[1])
    # Create day 0 portfolio
    portfolio = proc.create_portfolio(np.repeat(amount, N), stock_prices, fees, ledger)

    assert fma_window < sma_window, "Your SMA period must be less than your FMA period"
    for stock_id in range(N):
        # SMA calculated from day = sma_period
        sma = indi.moving_average(stock_prices[:,stock_id], window_size=sma_window) # size: N - sma_period
        # FMA calculated from day = fma_period
        fma = indi.moving_average(stock_prices[:,stock_id], window_size=fma_window)[sma_window-fma_window:] # size: N - sma_period
        # Comparison from day = sma_period - fma_period
        deltas = fma - sma 
        # Find buying and selling days
        cross_days = np.where((np.diff(np.sign(deltas)) != 0))[0] + 1

        for cross_day in cross_days:
            if (fma[cross_day - 1] > sma[cross_day - 1]):
                proc.sell(cross_day + sma_window-1, stock_id, stock_prices, fees, portfolio, ledger)
            else:
                proc.buy(cross_day + sma_window-1, stock_id, amount, stock_prices, fees, portfolio, ledger)

    # Sell final day stock
    sell_all_stock(stock_prices, fees, portfolio, ledger)

def momentum(stock_prices, osc_type='RSI', mom_period=7, cooldown_period=7, thresholds=(0.25, 0.75), amount=5000, fees=20, ledger='ledger_mom.txt'):
    '''
    Algorithmic trading strategy based on the use of oscillators.

    Input:
        stock_prices (ndarray): the stock price data
        osc_type (str, default RSI): oscillator to use (RSI or stochastic)
        mom_period (int, default 7): number of days used to calculate oscillator
        cooldown_period (int, default 7): number of days to wait between actions
        thresholds (tuple (len2), deafault (0.25, 0.75)): thresholds used to determine buying and selling days
        amount (float, default 5000): how much we spend on each purchase (must cover fees)
        fees (float, default 20): transaction fees
        ledger (str): path to the ledger file

    Output: 
        None
    '''
    # Number of stocks simulated
    N = int(stock_prices.shape[1])
    # Create day 0 portfolio
    portfolio = proc.create_portfolio(np.repeat(amount, N), stock_prices, fees, ledger)

    for stock_id in range(N):
        # Get oscillator values
        oscillator = indi.oscillator(stock_prices[:, stock_id], n=mom_period, osc_type=osc_type)
        # Set NaNs to 0
        nans_locs = np.isnan(oscillator)
        oscillator[nans_locs] = 0
        # Find buy days and sell days
        buy_days = np.where(oscillator < thresholds[0])[0]
        sell_days = np.where(oscillator > thresholds[1])[0]

        # Oscillator values are only valid when day >= mom_period
        day = mom_period 
        
        # Perform transactions with cooldown
        while day < len(stock_prices[:, stock_id]):
            if day in buy_days:
                proc.buy(day, stock_id, amount, stock_prices, fees, portfolio, ledger)
                day += cooldown_period 
            elif day in sell_days:
                proc.sell(day, stock_id, stock_prices, fees, portfolio, ledger)
                day += cooldown_period
            else:
                day += 1
        # Sell all stock on final day
        sell_all_stock(stock_prices, fees, portfolio, ledger)

def apply_all_strats(stock_prices, amount, fees, n_ran, sma_window, fma_window, stoch_period, rsi_period, thresholds, cooldown_period):
    '''
        Wrapper which calls all 4 strategy variants.
    '''
    random(stock_prices, n_ran, amount, fees, ledger='ledger_ran.txt')
    crossing_averages(stock_prices, sma_window, fma_window, amount, fees, ledger='ledger_cross.txt')
    momentum(stock_prices, 'RSI', rsi_period, cooldown_period, thresholds, amount, fees, ledger='ledger_rsi.txt')
    momentum(stock_prices, 'stochastic', stoch_period, cooldown_period, thresholds, amount, fees, ledger='ledger_stoch.txt')