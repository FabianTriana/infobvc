# Libraries:
import pandas as pd
import numpy as np
import requests
import seaborn as sns
import pandas_ta as ta
from datetime import date, datetime
from functools import reduce


# Data Info:
host = 'https://gist.githubusercontent.com/FabianTriana/'
key_quotes =  'b613ab453e7ac8a50b2561d48c2e9838'
key_fundamentals = 'dabd6fc124a0e6912d963365aaf69544'
key_non_publicly_traded_shares = 'dd9a9e2c22c55e1261980f45a9e5a324'
file_quotes = '/raw/infobvc_data.csv'
file_fundamentals = '/raw/infobvc_fundamental_data.csv'
file_non_publicly_traded_shares = '/raw/non_publicly_traded_shares.csv'
url_quotes = host + key_quotes + file_quotes
url_fundamentals = host + key_fundamentals + file_fundamentals
url_non_publicly_traded_shares = host + key_non_publicly_traded_shares + file_non_publicly_traded_shares

df_quotes = pd.read_csv(url_quotes)
df_quotes['date'] = pd.to_datetime(df_quotes['date'])
df_fundamentals = pd.read_csv(url_fundamentals)
df_non_publicly_traded_shares = pd.read_csv(url_non_publicly_traded_shares)

dict_data = {'quotes': df_quotes, 'fundamentals': df_fundamentals}


# Default Values:
minimum_date = str(date(date.today().year-1, 1, 1))
maximum_date = str(date.today())



# Available Tickers:
valid_tickers = list(pd.read_csv(url_quotes)['ticker'].unique())+['*']



# Valid aggregation periods:
valid_aggregation_periods = ['D', 'M', 'Q', 'Y']




# Functions:
# Get Quotes Function:
def get_quotes(tickers: list = ['*'], min_date: str = minimum_date, max_date: str = maximum_date, agg_period: str = 'd'):
  # Docstring:
  '''Get data of prices, quantities and volumes for stocks, period and aggregation period selected.

  Parameters
  ----------
  tickers : list
        List containing the tickers. Default value is ['*'] which means all available tickers.
  min_date: str
        Minimum date. It must have `%Y-%m-%d` format.
  max_date: str
        Maximum date. It must have `%Y-%m-%d` format.
  agg_period: str
        Aggregation period. The only valid values are `d` (day), `m` (month), `q` (quarter) and `y` (year).

  Returns
  ----------
  df : pandas.DataFrame
  '''
  # Capitalize all tickers selected:
  tickers = [the_tickers.upper() for the_tickers in tickers]

  # Uppercase for period selected:
  agg_period = agg_period.upper()

  # Check that all tickers selected are valid:
  not_valid_tickers = np.setdiff1d(tickers, valid_tickers)
  if len(not_valid_tickers) > 0:
    raise Exception('Error. {} are not valid tickers'.format(str(not_valid_tickers)))

  # Check that aggregation period is valid:
  if agg_period not in valid_aggregation_periods:
    raise Exception('Error. {} is not a valid aggregation period. Valid aggregation periods are {}'.format(agg_period, str(valid_aggregation_periods)))

  # Check that minimum date is a valid date:
  try:
    datetime.strptime(min_date, '%Y-%m-%d')
  except:
    raise Exception('''{} is not a valid date. Date must have '%Y-%m-%d' format'''.format(str(min_date)))

  # Check that maximum date is a valid date:
  try:
    datetime.strptime(max_date, '%Y-%m-%d')
  except:
    raise Exception('''{} is not a valid date. Date must have '%Y-%m-%d' format'''.format(str(max_date)))



  # Process Data:
  # Check if user has selected all tickers or only specific ones:
  if '*' in tickers:
    ticker_selection = valid_tickers
  else:
    ticker_selection = tickers
  min_date_filter = pd.to_datetime(df_quotes['date']) >= min_date
  max_date_filter = pd.to_datetime(df_quotes['date']) <= max_date
  ticker_filter = df_quotes['ticker'].isin(ticker_selection)
  df_filtered = df_quotes[min_date_filter & max_date_filter & ticker_filter].sort_values(by = ['ticker', 'date'], ascending = [True, False]).reset_index(drop = True)

  # Process according to aggregation period:
  if agg_period in ['D']:
    df = df_filtered
  elif agg_period in ['M', 'Q', 'Y']:
    df = df_filtered.groupby(['ticker', 'issuer', df_filtered['date'].dt.to_period(agg_period)], as_index = True).agg({'close': 'last', 'quantity': 'sum', 'open': 'first', 'low': 'min', 'high': 'max'}).reset_index().astype({'date': str})
  else:
    df  = pd.DataFrame({})

  # Return:
  return df



# Get Fundamentals Function:
def get_fundamentals(tickers: list = ['*'], ratios: bool = False):
  # Docstring:
  '''Get fundamental data for tickers selected.

  Parameters
  ----------
  ticker : list
        List containing the tickers. Default value is ['*'] which means all available tickers.
  ratios : bool
        If True then fundamental ratios (P/E, P/B and Yield) are also returned. If False then fundamental ratios are not returned.

  Returns
  ----------
  df = pandas.DataFrame
  '''
  # Capitalize all tickers selected:
  tickers = [the_tickers.upper() for the_tickers in tickers]
  df_quotes = dict_data['quotes']

  # Check that all tickers selected are valid:
  not_valid_tickers = np.setdiff1d(tickers, valid_tickers)
  if len(not_valid_tickers) > 0:
    raise Exception('Error. {} are not valid tickers'.format(str(not_valid_tickers)))



  # Process Data:
  # Check if user has selected all tickers or only specific ones:
  if '*' in tickers:
    ticker_selection = valid_tickers
  else:
    ticker_selection = tickers
  ticker_filter = df_fundamentals['ticker'].isin(ticker_selection)
  df_filtered = df_fundamentals[ticker_filter].reset_index(drop = True)
  df = df_filtered

  # Check if user has selected ratios:
  if ratios == True:
    df_quotes = df_quotes[df_quotes['close'].notna()]
    df_quotes = df_quotes.sort_values(by = ['date', 'ticker']).reset_index(drop = True)
    df_quotes_last = df_quotes.groupby(['issuer', 'ticker'], as_index = False).last()[['issuer', 'ticker', 'close']]
    df_quotes_last_selected = df_quotes_last[df_quotes_last['ticker'].isin(ticker_selection)]
    issuers = list(df['issuer'].unique())
    df = df[['ticker', 'dividend', 'sector', 'shares']].merge(df_quotes_last_selected, how = 'left', on = 'ticker')
    df['yield'] = df['dividend'] / df['close']


    # Get Data at Issuer Level:
    df_issuer = df_fundamentals[df_fundamentals['issuer'].isin(issuers)].reset_index(drop = True)
    # Correct for Davivienda NON PUBLICLY TRADED stocks:
    df_issuer['shares'] = np.where(df_issuer['issuer'] == 'BANCO DAVIVIENDA', df_issuer['shares'] + df_non_publicly_traded_shares[df_non_publicly_traded_shares['issuer'] == 'BANCO DAVIVIENDA']['shares'].unique()[0], df_issuer['shares'])

    df_issuer_group = df_issuer.groupby('issuer', as_index = False).agg({'shares': 'sum', 'sales': 'mean', 'net_income': 'mean', 'equity': 'mean', 'assets': 'mean'})
    df_issuer_group['book_value'] = df_issuer_group['equity'] / df_issuer_group['shares']
    df_issuer_group['eps'] = df_issuer_group['net_income'] / df_issuer_group['shares']
    df_issuer_group = df_issuer_group.drop(columns = ['shares'])

    # Get Issuer Capitalization:
    df_prices_issuer_cap = df_quotes_last[df_quotes_last['issuer'].isin(issuers)]
    df_prices_issuer_cap = df_prices_issuer_cap.merge(df_issuer[['ticker', 'issuer', 'shares']], how = 'left', on = ['ticker', 'issuer'])
    df_prices_issuer_cap['partial_cap'] = df_prices_issuer_cap['shares']*df_prices_issuer_cap['close']
    df_prices_issuer_cap_group = df_prices_issuer_cap.groupby('issuer', as_index = False).agg({'partial_cap': 'sum'}).rename(columns = {'partial_cap': 'cap'})

    # Merge dataframes:
    df = df.merge(df_issuer_group, how = 'left', on = 'issuer').merge(df_prices_issuer_cap_group, how = 'left', on = 'issuer')

    # Get ratios:
    df['price_to_book'] = df['close'] / df['book_value']
    df['price_to_earnings'] = df['close'] / df['eps']

    # Columns order:
    df = df[['ticker', 'shares',  'sales', 'net_income', 'equity',
             'assets', 'dividend', 'cap', 'yield', 'book_value',
             'eps', 'price_to_book', 'price_to_earnings', 'sector']]
  # Return:
  return df



# Get Technicals Function:
def get_technicals(tickers: list = ['*']):
  # Docstring:
  '''Get technical indicators (SMAs, EMAs, RSI) data for tickers selected.

  Parameters
  ----------
  tickers : list
        List containing the tickers. Default value is ['*'] which means all available tickers.

  Returns
  ----------
  df : pandas.DataFrame
  '''
  # Capitalize all tickers selected:
  df = get_quotes(tickers)
  # If there are NaN for close value, replace with average:
  df['close'] = np.where((df['close'].isna() | df['close'] <= 0), df['average'], df['close'])

  # Drop records where close value is zero:
  df = df[df['close'] > 0].reset_index(drop = True)

  # Order data by ticker and date:
  df = df.sort_values(by = ['ticker', 'date']).reset_index(drop = True)

  # Get technical indicators by ticker:
  # Avoid problems for tickers list containing only one value:
  dummy_value = ''
  if len(tickers) < 2:
    df_dummy = df.copy()
    df_dummy['ticker'] = df['ticker'] + '_dummy'
    dummy_value = list(df_dummy['ticker'].unique())[0]
    df = pd.concat([df, df_dummy], axis = 0).reset_index(drop = True)

  # Simple Moving Averages:
  df_sma_20 = df.groupby('ticker').apply(lambda x: ta.sma(x['close'], 20)).squeeze().reset_index().drop(columns = ['level_1'])
  df_sma_50 = df.groupby('ticker').apply(lambda x: ta.sma(x['close'], 50)).squeeze().reset_index().drop(columns = ['level_1', 'ticker'])
  df_sma_200 = df.groupby('ticker').apply(lambda x: ta.sma(x['close'], 200)).squeeze().reset_index().drop(columns = ['level_1', 'ticker'])

  # Exponential Moving Averages:
  df_ema_20 = df.groupby('ticker').apply(lambda x: ta.ema(x['close'], 20)).squeeze().reset_index().drop(columns = ['level_1', 'ticker'])
  df_ema_50 = df.groupby('ticker').apply(lambda x: ta.ema(x['close'], 50)).squeeze().reset_index().drop(columns = ['level_1', 'ticker'])
  df_ema_200 = df.groupby('ticker').apply(lambda x: ta.ema(x['close'], 200)).squeeze().reset_index().drop(columns = ['level_1', 'ticker'])

  # Relative Strength Index:
  df_rsi = df.groupby('ticker').apply(lambda x: ta.rsi(x['close'])).squeeze().reset_index().drop(columns = ['level_1', 'ticker'])

  # Merge Dataframes:
  dfs_technicals = [df_sma_20, df_sma_50, df_sma_200, df_ema_20, df_ema_50, df_ema_200, df_rsi]

  df_technical = reduce(lambda  left, right: pd.concat([left,right], axis = 1), dfs_technicals)

  # Drop dummy values:
  df_technical = df_technical[~df_technical['ticker'].str.contains('_dummy')].reset_index(drop = True)

  # Concat with dataframe:
  df = df[['date', 'ticker', 'issuer', 'close']]
  df = df[~df['ticker'].str.contains('_dummy')].reset_index(drop = True)
  df = pd.concat([df, df_technical], axis = 1)

  # Return:
  return df



# Plot Prices Function:
def plot_prices(tickers: list = ['*'], min_date: str = minimum_date, max_date: str = maximum_date, agg_period: str = 'd'):
  # Docstring:
  '''Plot data of prices for stocks, period and aggregation period selected.

  Parameters
  ----------
  tickers : list
        List containing the tickers. Default value is ['*'] which means all available tickers.
  min_date: str
        Minimum date. It must have `%Y-%m-%d` format.
  max_date: str
        Maximum date. It must have `%Y-%m-%d` format.
  agg_period: str
        Aggregation period. The only valid values are `d` (day), `m` (month), `q` (quarter) and `y` (year).

  Returns
  ----------
  plot : seaborn.axisgrid.FacetGrid
  '''
  df = get_quotes(tickers, min_date, max_date, agg_period)
  # If aggregation period is 'd' and there are NaN for close value, replace with average:
  if agg_period == 'd':
    df['close'] = np.where((df['close'].isna() | df['close'] <= 0), df['average'], df['close'])

  # Drop records where close value is zero:
  df = df[df['close'] > 0].reset_index(drop = True)
  plot = sns.relplot(data = df, x = 'date', y = 'close', row = 'ticker', kind = 'line', facet_kws={'sharey': False, 'sharex': True}, aspect = 2)
  return plot



# Comparative Plot Function:
def plot_comparative_prices(tickers: list = ['*'], min_date: str = minimum_date, max_date: str = maximum_date, agg_period: str = 'd'):
  # Docstring:
  '''Comparative plot of prices for stocks, period and aggregation period selected. Initial value is equivalent to 1 for every stock.

  Parameters
  ----------
  tickers : list
        List containing the tickers. Default value is ['*'] which means all available tickers.
  min_date: str
        Minimum date. It must have `%Y-%m-%d` format.
  max_date: str
        Maximum date. It must have `%Y-%m-%d` format.
  agg_period: str
        Aggregation period. The only valid values are `d` (day), `m` (month), `q` (quarter) and `y` (year).

  Returns
  ----------
  plot : seaborn.axisgrid.FacetGrid
  '''

  df = get_quotes(tickers, min_date, max_date, agg_period)

  # If aggregation period is 'd' and there are NaN for close value, replace with average:
  if agg_period == 'd':
    df['close'] = np.where((df['close'].isna() | df['close'] <= 0), df['average'], df['close'])

  # Sort by date, group by ticker and get initial value:
  df = df.sort_values(by = 'date').reset_index(drop = True)

  # Group by ticker and get initial value:
  df_group = df.groupby('ticker', as_index = False)['close'].first().rename(columns = {'close': 'initial_value'})

  # Merge with full dataframe:
  df = df.merge(df_group, how = 'left', on = 'ticker')

  # Get Relative Price:
  df['relative_price'] = df['close'] / df['initial_value']

  # Plot Relative Price:
  plot = sns.relplot(data = df, x = 'date', y = 'relative_price', hue = 'ticker', kind = 'line', aspect = 2)
  return plot



# Correlation Matrix Function:
def correlation_matrix(tickers: list = ['*'], min_date: str = minimum_date, max_date: str = maximum_date, agg_period: str = 'd', as_plot: bool = False, annot: bool = False, x_size : int = 10, y_size : int = 10):
  # Docstring:
  '''Correlation matrix of close prices for stocks, period and aggregation period selected.

  Parameters
  ----------
  tickers : list
        List containing the tickers. Default value is ['*'] which means all available tickers.
  min_date: str
        Minimum date. It must have `%Y-%m-%d` format.
  max_date: str
        Maximum date. It must have `%Y-%m-%d` format.
  agg_period: str
        Aggregation period. The only valid values are `d` (day), `m` (month), `q` (quarter) and `y` (year).

  Returns
  ----------
  df: pandas.DataFrame
  or
  plot : seaborn.axisgrid.FacetGrid
  '''
  df = get_quotes(tickers, min_date, max_date, agg_period)

  # If aggregation period is 'd' and there are NaN for close value, replace with average:
  if agg_period == 'd':
    df['close'] = np.where((df['close'].isna() | df['close'] <= 0), df['average'], df['close'])

  # Get Correlation Matrix:
  df_corr = df.pivot_table(index= 'date', columns = 'ticker', values = 'close', aggfunc = 'mean').corr()
  result = df_corr

  # If as_plot is True then return correlation matrix plot:
  if as_plot == True:
    fig, ax = plt.subplots(figsize = (x_size, y_size))
    sns.heatmap(df_corr, vmin = -1, vmax = 1, cmap = 'RdYlGn', center = 0, annot = annot, ax = ax)
    ax.set(title = 'Correlation Matrix')
    result = ax

  # Return:
  return result
