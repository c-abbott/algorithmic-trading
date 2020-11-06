# Evaluate performance.
import numpy as np
import pandas as pd
import os.path as path

def read_ledger(ledger_file):
    '''
    Reads and reports useful information from ledger_file.
    '''
     # Locating relevant file
    BASE_PATH = path.dirname(__file__)
    FILE_PATH = path.abspath(path.join(BASE_PATH, "..", ledger_file))
    # Read ledger in as pandas dataframeper
    df = pd.read_csv(FILE_PATH, sep=',', header=None, names=['transaction', 'date', 'stock', 'num_shares', 'price', 'net_capital'])
    # Determine number of transactions
    num_bought = round(len(df[df.transaction == 'buy']), 2)
    num_sold = round(len(df[df.transaction == 'sell']), 2)
    # Summary stats
    money_spent = round(np.abs(np.sum(df[df.transaction == 'buy']['net_capital'])), 2)
    money_earnt = round(np.abs(np.sum(df[df.transaction == 'sell']['net_capital'])), 2)
    # Determine net change in capital
    total_net_captial = money_earnt - money_spent
    # Printing summary
    print(f'Days simulated: {np.max(df.date)-np.min(df.date)} \n' + \
            f'Total number of transactions: {num_bought + num_sold} \n' + \
            f'Total spent: £{money_spent} \n' + \
            f'Total earnt: £{money_earnt} \n' + \
            f'Net capital: £{total_net_captial} \n')
            