# Functions to implement our trading strategy.
import numpy as np
import process as proc
import data
import indicators as indi

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
    for i in range(N):
        proc.sell(stock_prices.shape[0]-1, i, stock_prices, fees, portfolio, ledger)

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
    portfolio = proc.create_portfolio(np.ones(N)*amount, stock_prices, fees, ledger)

    # Determine number of actions and which days they occur
    action_freq = int(np.floor(stock_prices.shape[0] / period))
    # Determine dates on which we act
    action_days = [(period*i-1) for i in range(1, action_freq+2)]

    # Loop over all actions
    for i in range(action_freq):
        # Loop over action for each stock
        for j in range(N):
            # Randomly sample action from discrete unif dist.
            action = np.random.choice([0, 1, 2])

            if action == 1: # Buying
                proc.buy(action_days[i], j, amount, stock_prices, fees, portfolio, ledger)
            elif action == 2: # Selling
                proc.sell(action_days[i], j, stock_prices, fees, portfolio, ledger)
    
    # Finish with selling all stock on final day
    for k in range(N):
        proc.sell(stock_prices.shape[0]-1, k, stock_prices, fees, portfolio, ledger)

def crossing_averages(stock_prices, sma_period=200, fma_period=50, amount=5000, fees=20, ledger='ledger_cross.txt'):
    '''
    Algorithmic trading strategy based of crossing of the slow moving average (SMA) and fast
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
    portfolio = proc.create_portfolio(np.ones(N)*amount, stock_prices, fees, ledger)

    assert fma_period < sma_period, "Your SMA period must be less than your FMA period"
    for i in range(N):
        # SMA calculated from day = sma_period
        sma = indi.moving_average(stock_prices[:,i], n=sma_period) # size: N - sma_period
        # FMA calculated from day = fma_period
        fma = indi.moving_average(stock_prices[:,i], n=fma_period) # size: N - fma_period, starts on day fma_period
        # Comparison from day = sma_period - fma_period
        delta = fma[sma_period-fma_period:] - sma 
        # Find buying and selling days
        cross_days = np.where((np.diff(np.sign(delta)) != 0))[0] + (sma_period - fma_period + 1)
        
        # Find initial state of FMA relatie to SMA
        is_fma_negative = np.sign(delta[i]) < 0

        # Buying and selling
        for j in range(len(cross_days)):
            if is_fma_negative:
                proc.buy(cross_days[j], i, amount, stock_prices, fees, portfolio, ledger)
                is_fma_negative = False
            else:
                proc.sell(cross_days[j], i, stock_prices, fees, portfolio, ledger)
                is_fma_negative = True
        
    # Sell final day stock
    sell_all_stock(stock_prices, fees, portfolio, ledger)

        
if __name__ == "__main__":
    stock_prices = data.get_data()
    crossing_averages(stock_prices)
    random(stock_prices)