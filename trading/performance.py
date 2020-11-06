# Evaluate performance.
import numpy as np
import pandas as pd
import os.path as path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from trading import process as proc
from trading import indicators as indi

def read_ledger(ledger, print_flag=True):
    '''
    Reads and reports useful information from ledger_file.
    Input:
        ledger (str): path to the ledger file
        print_flag (bool): flag to determine whether to print summary statistics

    Output:
        total_net_captial (float): total net change in capital after applying strategies
    '''
    # Locating relevant file
    BASE_PATH = path.dirname(__file__)
    FILE_PATH = path.abspath(path.join(BASE_PATH, "..", ledger))
    # Read ledger in as pandas dataframeper
    df = pd.read_csv(FILE_PATH, sep=',', header=None, names=['transaction', 'date', 'stock', 'num_shares', 'price', 'net_capital'])
    # Determine number of transactions
    num_bought = round(len(df[df.transaction == 'buy']), 2)
    num_sold = round(len(df[df.transaction == 'sell']), 2)
    # Summary stats
    money_spent = round(np.abs(np.sum(df[df.transaction == 'buy']['net_capital'])), 2)
    money_earnt = round(np.abs(np.sum(df[df.transaction == 'sell']['net_capital'])), 2)
    # Determine net change in capital
    total_net_captial = round(money_earnt - money_spent, 2)
    # Printing summary
    if print_flag:
        print(f'Stocks Stimulated: {np.max(df.stock) + 1} \n' + \
                f'Days simulated: {np.max(df.date)-np.min(df.date)} \n' + \
                f'Total number of transactions: {num_bought + num_sold} \n' + \
                f'Total spent: £{money_spent} \n' + \
                f'Total earnt: £{money_earnt} \n' + \
                f'Net capital: £{total_net_captial} \n')
    else:
        return total_net_captial

def plot_net_capital(days):
    '''
    Plotting function to visualise the total change in capital when implmenting
    a particular trading strategy

    Input:
        days (float): number of days to produce plot for

    Output: None
    '''
    # Plot titles and relevant files
    strats = ['Random', 'Crossing Averages', 'Stochastic Momentum', 'RSI Momentum']
    ledgers = ['ledger_ran.txt', 'ledger_cross.txt', 'ledger_stoch.txt', 'ledger_rsi.txt']
    
    # Create and format figure
    fig, axes = plt.subplots(2, 2, sharex = True, figsize=(16, 10))
    positions = [(0,0), (0,1), (1,0), (1,1)]
    plt.setp(axes, xlim=(0, days))
    plt.setp(axes[-1, :], xlabel='Days')
    plt.setp(axes[:, 0], ylabel='Net Capital')

    # Formatting legend
    loss_patch = mpatches.Patch(color='red', label='Loss')
    profit_patch = mpatches.Patch(color='green', label='Profit')
    fig.legend(handles=[loss_patch, profit_patch], ncol=2, loc='lower center', fontsize=15)

    # Days to produce plot for
    days = np.arange(0, days)

    for strat, ledger in enumerate(ledgers):
        # Storage
        totals = np.zeros_like(days)
        df = pd.read_csv(ledger, sep=',', header=None, names=['transaction', 'date', 'stock', 'num_shares', 'price', 'net_capital'])
        # Replace NaNs with 0
        df.fillna(value = 0, inplace=True)
        # Store total money spent/earnt on each day
        for _, row in df.iterrows():
            totals[row['date']] += row['net_capital']
        # Track net capital over time
        cum_sum = np.cumsum(totals)

        # Subplot plotting
        axes[positions[strat]].set_title(strats[strat])
        axes[positions[strat]].plot(days, cum_sum, color = 'black', label = 'Net Capital')
        axes[positions[strat]].fill_between(x=days, y1=np.zeros_like(days), y2=cum_sum, where=cum_sum <= 0, facecolor = "red")
        axes[positions[strat]].fill_between(x=days, y1=np.zeros_like(days), y2=cum_sum, where=cum_sum > 0, facecolor = "green")


def plot_indicators(stock_price, days, sma_window, fma_window, rsi_period, stoch_period):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(20, 10))
    plt.setp(axes, xlim=(sma_window, days))

    # Getting smas and fmas
    smas = indi.moving_average(stock_price, window_size=sma_window)
    fmas = indi.moving_average(stock_price, window_size=fma_window)[sma_window-fma_window:]
    # Comparison from day = sma_period - fma_period
    deltas = fmas - smas
    # Find buying and selling days
    cross_days = np.where((np.diff(np.sign(deltas)) != 0))[0] + 1
    ma_sell_days = []
    ma_buy_days = []
    for cross_day in cross_days:
            if (fmas[cross_day - 1] > smas[cross_day - 1]):
                ma_sell_days.append(cross_day + sma_window)
            else:
                ma_buy_days.append(cross_day + sma_window)

    # Computing oscillator values
    rsis = indi.oscillator(stock_price, n=rsi_period, osc_type='rsi')[sma_window-1:]
    stochs = indi.oscillator(stock_price, n=stoch_period, osc_type='stochastic')[sma_window-1:]

    # Plotting MA Comparison
    days = np.arange(sma_window, days+1)
    axes[0].set_title('MA Comparison')
    for buy_day in ma_buy_days:
        axes[0].axvline(x=buy_day, color='g', linestyle='--')
    for sell_day in ma_sell_days:
        axes[0].axvline(x=sell_day, color='r', linestyle='--')
    axes[0].plot(days, stock_price[sma_window-1:], 'y-', label = 'Stock Price')
    axes[0].plot(days, smas, 'c-', label = f'{sma_window}-day SMA')
    axes[0].plot(days, fmas, 'm-', label=f'{fma_window}-day FMA')
    axes[0].legend()

    # Plotting Oscillator Comparison
    axes[1].set_title('Oscillator Comparison')
    axes[1].plot(days, rsis, 'r-', label = f'{rsi_period}-day RSI')
    axes[1].plot(days, stochs, 'b-', label=f'{stoch_period}-day Stochastic')
    axes[1].legend()