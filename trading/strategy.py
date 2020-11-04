# Functions to implement our trading strategy.
import numpy as np
from trading import process as proc
from trading import data as data

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
    N = stock_prices.shape[1] 
    # Create day 0 portfolio
    portfolio = proc.create_portfolio(np.ones(N)*amount, stock_prices, fees, ledger)

    # Determine number of actions and which days they occur
    action_freq = int(np.floor(stock_prices.shape[0] / period) - 1)
    action_days = [period*i for i in range(1, action_freq+1)]

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