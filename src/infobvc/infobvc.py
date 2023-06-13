# Libraries:
import pandas as pd
import numpy as np
import requests
import seaborn as sns
from datetime import date, datetime


# API Info:
# API URL:
url_tickers = "https://testing_infobvc_render.p.rapidapi.com/tickers"
url_quotes = "https://testing_infobvc_render.p.rapidapi.com/quotes"
url_fundamentals = "https://testing_infobvc_render.p.rapidapi.com/fundamentals" 

# API Headers:
headers = {
	"X-RapidAPI-Key": "3dafbd3f4cmsh991a03a41685a07p1ee876jsn71d2eb5d7729",
	"X-RapidAPI-Host": "testing_infobvc_render.p.rapidapi.com",
	"Content-Type": "application/json"
}


# Default Values:
minimum_date = str(date(date.today().year-1, 1, 1))
maximum_date = str(date.today())

# Available Tickers:
df_tickers = pd.DataFrame(requests.request("GET", url_tickers, headers = headers).json())
valid_tickers = list(df_tickers['tickers'].unique())+['*']

# Valid aggregation periods:
valid_aggregation_periods = ['d', 'm', 'q', 'y']

# Functions:
# Get Quotes Function:
def get_quotes(ticker: list = ['*'], min_date: str = minimum_date, max_date: str = maximum_date, agg_period: str = 'd'):
  # Docstring:
  '''Get data of prices, quantities and volumes for stocks, period and aggregation period selected.
  
  Parameters
  ----------
  ticker : list
        List containing the tickers. Default value is ['*'] which means all available tickers.
  min_date: str
        Minimum date. It must have `%Y-%m-%d` format.
  max_date: str
        Maximum date. It must have `%Y-%m-%d` format.
  agg_period: str
        Aggregation period. The only valid values are `d` (day), `m` (month), `q` (quarter) and `y` (year).
  
  Returns
  ----------
  df : pd.DataFrame
  '''
  # Capitalize all tickers selected:
  ticker = [the_ticker.upper() for the_ticker in ticker]

  # Lowercase for period selected:
  agg_period = agg_period.lower()

  # Check that all tickers selected are valid:
  not_valid_tickers = np.setdiff1d(ticker, valid_tickers) 
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

  # Payload based on input values:
  payload = {"ticker": ticker,
             "min_date": min_date,
             "max_date": max_date,
             "aggregation_period": agg_period}
  response = requests.request("POST", url_quotes, json = payload, headers = headers)
  df = pd.DataFrame(response.json())
  df['date'] = pd.to_datetime(df['date'])
  return df



# Get Fundamentals Function:
def get_fundamentals(ticker: list = ['*'], ratios: bool = False):
  # Docstring:
  '''Get fundamental data for tickers selected.
  
  Parameters
  ----------
  ticker : list
        List containing the tickers. Default value is ['*'] which means all available tickers.
  ratios : bool
        If True then fundamental ratios (P/E, P/B and Yield) are also returned. If False then fundamental ratios are not returned
  
  Returns
  ----------
  df = pd.DataFrame
  '''
  # Payload based on input values:
  payload = {"ticker": ticker}
  response = requests.request("POST", url_fundamentals, json = payload, headers = headers)
  df = pd.DataFrame(response.json())
  return df



# Plot Prices Function:
def plot_prices(ticker: list = ['*'], min_date: str = minimum_date, max_date: str = maximum_date, agg_period: str = 'd'):
  df = get_quotes(ticker, min_date, max_date, agg_period)
  plot = sns.relplot(data = df, x = 'date', y = 'close', row = 'ticker', kind = 'line', facet_kws={'sharey': False, 'sharex': True}, aspect = 2)
  return plot



# Comparative Plot Function:
def comparative_plot(ticker: list = ['*'], min_date: str = minimum_date, max_date: str = maximum_date, agg_period: str = 'd'):
  df = get_quotes(ticker, min_date, max_date, agg_period)
  
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
