import numpy as np
import csv

# Functions to process transactions.
def log_transaction(transaction_type, date, stock, number_of_shares, price, fees, ledger_file):
    '''
    Record a transaction in the file ledger_file. If the file doesn't exist, create it.
    
    Input:
        transaction_type (str): 'buy' or 'sell'
        date (int): the date of the transaction (nb of days since day 0)
        stock (int): the stock we buy or sell (the column index in the data array)
        number_of_shares (int): the number of shares bought or sold
        price (float): the price of a share at the time of the transaction
        fees (float): transaction fees (fixed amount per transaction, independent of the number of shares)
        ledger_file (str): path to the ledger file
    
    Output: returns None.
        Writes one line in the ledger file to record a transaction with the input information.
        This should also include the total amount of money spent (negative) or earned (positive)
        in the transaction, including fees, at the end of the line.
        All amounts should be reported with 2 decimal digits.
    '''
    # Determining money gained or spent
    delta_money = 0
    if transaction_type == 'buy':
        delta_money  -= number_of_shares * price + fees
    elif transaction_type == 'sell':
        delta_money += number_of_shares * price - fees

    # Info to written to file
    transaction_info = [transaction_type, date, stock, number_of_shares, price, delta_money]

    # Coverting items to strings and 2 d.p. so they can be written to file
    for item in transaction_info:
        if type(item) == str:
            pass
        else:
            item = str(round(item, 2))

    # File writer
    with open(ledger_file, 'a+') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(transaction_info)

def buy(date, stock, available_capital, stock_prices, fees, portfolio, ledger_file):
    '''
    Buy shares of a given stock, with a certain amount of money available.
    Updates portfolio in-place, logs transaction in ledger.
    
    Input:
        date (int): the date of the transaction (nb of days since day 0)
        stock (int): the stock we want to buy
        available_capital (float): the total (maximum) amount to spend,
            this must also cover fees
        stock_prices (ndarray): the stock price data
        fees (float): total transaction fees (fixed amount per transaction)
        portfolio (list): our current portfolio
        ledger_file (str): path to the ledger file
    
    Output: None
    '''
    # Retrieve stock price
    price = stock_prices[int(date), int(stock)]
    # Buy as many shares as possible
    number_of_shares = np.floor( (available_capital - fees) / price )
    # Update portfolio
    portfolio[stock] += number_of_shares
    # Log transaction
    log_transaction('buy', date, stock, number_of_shares, price, fees, ledger_file)

def sell(date, stock, stock_prices, fees, portfolio, ledger_file):
    '''
    Sell all shares of a given stock.
    Updates portfolio in-place, logs transaction in ledger.
    
    Input:
        date (int): the date of the transaction (nb of days since day 0)
        stock (int): the stock we want to sell
        stock_prices (ndarray): the stock price data
        fees (float): transaction fees (fixed amount per transaction)
        portfolio (list): our current portfolio
        ledger_file (str): path to the ledger file
    
    Output: None
    '''
    # Retrieve stock price
    price = stock_prices[int(date), int(stock)]
    # Sell all shares of stock and update portfolio
    number_of_shares = portfolio[stock]
    # We can only sell the stock if we own it!
    if number_of_shares != 0:
        portfolio[stock] = 0
        # Log transaction
        log_transaction('sell', date, stock, number_of_shares, price, fees, ledger_file)



def create_portfolio(available_amounts, stock_prices, fees, ledger_file):
    '''
    Create a portfolio by buying a given number of shares of each stock.
    
    Input:
        available_amounts (list): how much money we allocate to the initial
            purchase for each stock (this should cover fees)
        stock_prices (ndarray): the stock price data
        fees (float): transaction fees (fixed amount per transaction)
        ledger_file (str): path to the ledger file
    
    Output:
        portfolio (list): our initial portfolio
    '''
    # Define number of stocks, N, and date
    N = stock_prices.shape[1]
    date = 0
    # Create portfolio
    portfolio = np.zeros(N).tolist()
    # Populating day 0 portfolio
    for i in range(N):
        # iterator i represents the stock to buy
        buy(date, i, available_amounts[i], stock_prices, fees, portfolio, ledger_file)
    return portfolio