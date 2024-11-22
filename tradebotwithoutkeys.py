import os
import pandas as pd
import numpy as np
import warnings
import ccxt.async_support as ccxt
import asyncio
import logging
from google.cloud import storage
from ta.trend import EMAIndicator, ADXIndicator, MACD
from ta.momentum import RSIIndicator, WilliamsRIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from sklearn.model_selection import train_test_split
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import openai
import json
import asyncio
import sys
from sklearn.model_selection import GridSearchCV
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from datetime import datetime, timedelta
import pickle
import backtrader as bt
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from transformers import pipeline
from sklearn.model_selection import TimeSeriesSplit
import prometheus_client as prom
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import nest_asyncio
import time
from scipy.stats import linregress
from prometheus_client import Gauge, start_http_server

nest_asyncio.apply()


if sys.platform.startswith('win'):
	asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Constants
MAX_CONCURRENT_TASKS = 10
PROMETHEUS_PORT = 8000





from sklearn.feature_selection import VarianceThreshold

#Remove Constant Features
from sklearn.feature_selection import VarianceThreshold
def remove_constant_features(data, threshold=0.01):
	"""Remove features with low variance (constant features)."""
	try:
		# Apply VarianceThreshold to remove constant features
		selector = VarianceThreshold(threshold)
		filtered_data = selector.fit_transform(data)
		selected_columns = data.columns[selector.get_support()]
		return pd.DataFrame(filtered_data, columns=selected_columns)
	except Exception as e:
		logging.error(f"Error in removing constant features: {e}")
		return data


#Handle Missing Values
def handle_missing_values(data, method='mean'):
	"""Handle missing values in the dataset using the specified method."""
	try:
		if method == 'mean':
			return data.fillna(data.mean())
		elif method == 'median':
			return data.fillna(data.median())
		elif method == 'drop':
			return data.dropna()
		else:
			raise ValueError("Invalid method for handling missing values.")
	except Exception as e:
		logging.error(f"Error handling missing values: {e}")
		return data


#Balance Data
from imblearn.over_sampling import SMOTE
def balance_data(features, target):
	"""Balance the dataset using SMOTE."""
	try:
		smote = SMOTE(random_state=42)
		balanced_features, balanced_target = smote.fit_resample(features, target)
		return balanced_features, balanced_target
	except Exception as e:
		logging.error(f"Error in balancing data: {e}")
		return features, target


#Initialize LGBM Model
from lightgbm import LGBMClassifier
def initialize_lgbm_model():
	"""Initialize the LightGBM model with adjusted learning rate and estimators."""
	model = LGBMClassifier(
		max_depth=5,               # Depth of the trees
		learning_rate=0.05,        # Adjusted learning rate
		n_estimators=200,          # Increased number of estimators
		class_weight='balanced'    # Handle imbalanced classes
	)
	return model


#Normalize Data
from sklearn.preprocessing import StandardScaler
def normalize_data(data):
	"""Normalize data using StandardScaler."""
	try:
		scaler = StandardScaler()
		scaled_data = scaler.fit_transform(data)
		return pd.DataFrame(scaled_data, columns=data.columns)
	except Exception as e:
		logging.error(f"Error normalizing data: {e}")
		return data



def calculate_regression_trend(df, window=20):
	"""
	Calculate linear regression trends for the 'close' prices in the DataFrame.

	Args:
		df (pd.DataFrame): Market data with a 'close' column.
		window (int): Rolling window size for regression trend calculation.

	Returns:
		pd.DataFrame: DataFrame with added regression slope and intercept columns.
	"""
	slopes = []
	intercepts = []

	for i in range(len(df)):
		if i < window - 1:
			slopes.append(np.nan)
			intercepts.append(np.nan)
		else:
			y = df['close'].iloc[i - window + 1 : i + 1]
			x = np.arange(window)
			slope, intercept, _, _, _ = linregress(x, y)
			slopes.append(slope)
			intercepts.append(intercept)

	df['regression_slope'] = slopes
	df['regression_intercept'] = intercepts
	return df

def strategy_fn_with_regression(df):
	"""
	Strategy function using regression trends to filter trades.

	Args:
		df (pd.DataFrame): DataFrame with market data and regression trend calculations.

	Returns:
		str: A trade signal ('buy', 'sell', or 'hold').
	"""
	if len(df) < 50 or 'regression_slope' not in df.columns:
		return "hold"

	latest_slope = df['regression_slope'].iloc[-1]
	latest_close = df['close'].iloc[-1]
	intercept = df['regression_intercept'].iloc[-1]
	trendline_value = intercept + latest_slope * (len(df) - 1)

	if latest_slope > 0 and latest_close > trendline_value:
		return "buy"

	if latest_slope < 0 and latest_close < trendline_value:
		return "sell"

	return "hold"

async def enhanced_backtest_with_regression(df, initial_balance, fee=0.001):
	"""
	Backtesting with regression trends integrated.

	Args:
		df (pd.DataFrame): Historical market data with regression trend calculations.
		initial_balance (float): Starting balance.
		fee (float): Trading fee percentage.

	Returns:
		dict: Backtesting results including final balance and metrics.
	"""
	df = calculate_regression_trend(df)
	return await enhanced_backtest_strategy(df, strategy_fn_with_regression, initial_balance, fee)

import logging
import pandas as pd

async def get_live_balance(asset="USDT"):
    """
    Fetches the live balance for a specific asset from the Binance account.

    Args:
        asset (str): The asset symbol to check balance for (default is "USDT").

    Returns:
        float: The balance of the specified asset.
    """
    try:
        account_info = binance_client.futures_account_balance()

        for balance in account_info:
            if balance['asset'] == asset:
                logging.info(f"Fetched live balance for {asset}: {balance['balance']}")
                return float(balance['balance'])

        logging.warning(f"Asset {asset} not found in account balances.")
        return 0.0
    except Exception as e:
        logging.error(f"Error fetching live balance for {asset}: {e}")
        return 0.0

import logging
import pandas as pd

#Run Backtest Experiments
async def run_backtest_experiments(symbols, timeframes, initial_balance):
    """
    Superior backtesting experiment runner with multiple strategies and enhanced error handling.

    Args:
        symbols (list): List of trading symbols to backtest.
        timeframes (list): List of timeframes for historical data.
        initial_balance (float): Initial balance for backtesting.

    Returns:
        pd.DataFrame: DataFrame with backtesting results.
    """
    try:
        results = []

        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    logging.info(f"Fetching data for {symbol} on {timeframe}...")
                    
                    # Fetch historical data
                    data = await fetch_and_save_historical_data(symbol, timeframe)

                    if data is None or data.empty:
                        logging.warning(f"No data found for {symbol} on {timeframe}. Skipping.")
                        continue

                    # Prepare the data for backtesting
                    logging.info(f"Preparing data for {symbol} on {timeframe}...")
                    data = prepare_data(data)

                    # Run the original strategy
                    logging.info(f"Running original strategy for {symbol} on {timeframe}...")
                    original_result = await enhanced_backtest_strategy(data, strategy_fn, initial_balance)
                    if not original_result.get("trades"):
                        logging.warning(f"No trades executed for {symbol} on {timeframe} using the original strategy.")
                        original_result = {"total_pnl": 0, "trades": []}

                    # Add regression trend and run the regression-based strategy
                    logging.info(f"Calculating regression trends for {symbol} on {timeframe}...")
                    data_with_trend = calculate_regression_trend(data)
                    logging.info(f"Running regression-based strategy for {symbol} on {timeframe}...")
                    regression_result = await enhanced_backtest_strategy(
                        data_with_trend, strategy_fn_with_regression, initial_balance
                    )
                    if not regression_result.get("trades"):
                        logging.warning(f"No trades executed for {symbol} on {timeframe} using the regression strategy.")
                        regression_result = {"total_pnl": 0, "trades": []}

                    # Store results for both strategies
                    results.extend([
                        {
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "strategy": "Original",
                            "total_pnl": original_result.get("total_pnl", 0),
                            "trades": original_result.get("trades", [])
                        },
                        {
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "strategy": "Regression",
                            "total_pnl": regression_result.get("total_pnl", 0),
                            "trades": regression_result.get("trades", [])
                        }
                    ])

                except Exception as e:
                    logging.error(f"Error during backtesting for {symbol} on {timeframe}: {e}")

        # Check if results are empty
        if not results:
            logging.error("No backtesting results generated.")
            return pd.DataFrame()

        # Convert results to DataFrame
        logging.info("Backtesting completed. Generating results DataFrame...")
        return pd.DataFrame(results)

    except Exception as e:
        logging.error(f"Unexpected error in run_backtest_experiments: {e}")
        return pd.DataFrame()





async def live_trading_with_regression(symbols, initial_balance):
	"""
	Live trading loop with regression trends.

	Args:
		symbols (list): List of symbols to trade.
		initial_balance (float): Starting balance for live trading.
	"""
	for symbol in symbols:
		market_data = await fetch_and_save_historical_data(symbol, '1h')
		if market_data is None:
			continue

		market_data = calculate_regression_trend(market_data)
		signal = strategy_fn_with_regression(market_data)

		if signal == "buy":
			logging.info(f"Buy signal for {symbol} at {market_data['close'].iloc[-1]}")
			# Execute buy order
		elif signal == "sell":
			logging.info(f"Sell signal for {symbol} at {market_data['close'].iloc[-1]}")
			# Execute sell order


from prometheus_client import Gauge

regression_slope_metric = Gauge("regression_slope", "Latest regression slope for a symbol")
regression_intercept_metric = Gauge("regression_intercept", "Latest regression intercept for a symbol")

async def update_regression_metrics(symbol, data):
	"""
	Update Prometheus metrics for regression trends.

	Args:
		symbol (str): Symbol being monitored.
		data (pd.DataFrame): Market data with regression trends.
	"""
	if "regression_slope" in data.columns:
		latest_slope = data["regression_slope"].iloc[-1]
		latest_intercept = data["regression_intercept"].iloc[-1]
		regression_slope_metric.labels(symbol=symbol).set(latest_slope)
		regression_intercept_metric.labels(symbol=symbol).set(latest_intercept)
	else:
		logging.warning(f"Regression data missing for {symbol}.")

async def strategy_fn_with_refinement(strategy_fn):
	"""
	Periodically refine the strategy using GPT-based feedback.
	"""
	try:
		# Fetch current strategy details
		refined_strategy = await refined_strategy_with_gpt_retry(strategy_fn)
		logging.info(f"GPT-based refinement completed. Updated strategy: {refined_strategy}")
		return refined_strategy
	except Exception as e:
		logging.error(f"Error during GPT-based strategy refinement: {e}")
		return strategy_fn  # Fallback to original strategy

#Fibonacci Strategy
def strategy_fn(df):
    """
    Advanced strategy leveraging Fibonacci retracement levels, EMA crossovers,
    regression trends, and volatility-based filters.

    Args:
        df (pd.DataFrame): DataFrame containing historical price data with indicators.

    Returns:
        str: A trade signal ('buy', 'sell', or 'hold').
    """
    # Ensure enough data for reliable calculations
    if len(df) < 50:
        logging.warning("Insufficient data for strategy calculations. Returning 'hold'.")
        return "hold"

    # Verify required columns exist in the DataFrame
    required_columns = [
        'fib_23_6', 'fib_38_2', 'fib_50', 'fib_61_8', 
        'EMA_20', 'EMA_50', 'close', 'regression_slope', 
        'ATR', 'high', 'low'
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.warning(f"Missing required columns for strategy: {missing_columns}. Returning 'hold'.")
        return "hold"

    try:
        # Fetch the latest data
        fib_23_6, fib_38_2, fib_50, fib_61_8 = (
            df['fib_23_6'].iloc[-1],
            df['fib_38_2'].iloc[-1],
            df['fib_50'].iloc[-1],
            df['fib_61_8'].iloc[-1]
        )
        close_price = df['close'].iloc[-1]
        ema_20 = df['EMA_20'].iloc[-1]
        ema_50 = df['EMA_50'].iloc[-1]
        regression_slope = df['regression_slope'].iloc[-1]
        atr = df['ATR'].iloc[-1]

        # EMA crossover signal
        ema_crossover = ema_20 > ema_50  # True for bullish, False for bearish

        # Regression trend alignment
        bullish_trend = regression_slope > 0
        bearish_trend = regression_slope < 0

        # Volatility filter to avoid high-risk trades
        if atr > close_price * 0.05:  # Example threshold: 5% volatility
            logging.warning("High volatility detected. Skipping trade. Returning 'hold'.")
            return "hold"

        # Long signal: EMA crossover, bullish trend, and price above Fibonacci support
        if ema_crossover and bullish_trend and (fib_61_8 < close_price < fib_50 or close_price > fib_50):
            logging.info(f"Buy signal triggered: Close={close_price}, EMA_20={ema_20}, EMA_50={ema_50}, Fib_61.8={fib_61_8}, Fib_50={fib_50}.")
            return "buy"

        # Short signal: No EMA crossover, bearish trend, and price near Fibonacci resistance
        if not ema_crossover and bearish_trend and (fib_23_6 > close_price > fib_38_2 or close_price < fib_38_2):
            logging.info(f"Sell signal triggered: Close={close_price}, EMA_20={ema_20}, EMA_50={ema_50}, Fib_23.6={fib_23_6}, Fib_38.2={fib_38_2}.")
            return "sell"

        # No clear signal; hold position
        logging.debug(f"No trade signal: Close={close_price}, EMA_20={ema_20}, EMA_50={ema_50}. Returning 'hold'.")
        return "hold"

    except Exception as e:
        logging.error(f"Error in strategy_fn: {e}. Returning 'hold'.")
        return "hold"




	

# Prepare Data
import pandas as pd

def prepare_data(df):
	"""
	Prepares the DataFrame by calculating required indicators.

	Args:
		df (pd.DataFrame): Original historical data.

	Returns:
		pd.DataFrame: DataFrame with added indicators.
	"""
	# Calculate EMA
	df['EMA_20'] = df['close'].ewm(span=20).mean()
	df['EMA_50'] = df['close'].ewm(span=50).mean()

	# Calculate Fibonacci levels
	high = df['high'].max()
	low = df['low'].min()
	df['fib_23_6'] = high - (high - low) * 0.236
	df['fib_38_2'] = high - (high - low) * 0.382
	df['fib_50'] = high - (high - low) * 0.5
	df['fib_61_8'] = high - (high - low) * 0.618

	return df


#Fetch All Symbols To Speed Up
async def fetch_all_symbols_data(symbols, timeframes):
	"""
	Fetch historical data for all symbols across all timeframes in parallel.

	Args:
		symbols (list): List of symbols to fetch data for.
		timeframes (list): List of timeframes to fetch data for.

	Returns:
		dict: Dictionary containing data for each symbol and timeframe.
	"""
	tasks = [
		fetch_and_save_historical_data(symbol, timeframe)
		for symbol in symbols
		for timeframe in timeframes
	]
	results = await asyncio.gather(*tasks, return_exceptions=True)

	data = {}
	for idx, task_result in enumerate(results):
		symbol = symbols[idx // len(timeframes)]
		timeframe = timeframes[idx % len(timeframes)]
		if isinstance(task_result, Exception):
			logging.warning(f"Failed to fetch data for {symbol} ({timeframe}): {task_result}")
		else:
			if symbol not in data:
				data[symbol] = {}
			data[symbol][timeframe] = task_result

	return data

async def analyze_in_batches(symbols, strategy_fn, batch_size=5):
	"""
	Analyze symbols in batches using the analyze_symbols function.

	Args:
		symbols (list): List of symbols to analyze.
		strategy_fn (callable): Strategy function to apply to each symbol.
		batch_size (int): Number of symbols to analyze in each batch.

	Returns:
		list: List of trade opportunities from analyzed symbols.
	"""
	trade_opportunities = []
	for i in range(0, len(symbols), batch_size):
		batch = symbols[i:i + batch_size]
		try:
			batch_results = await analyze_symbols(batch, strategy_fn)
			trade_opportunities.extend(batch_results)
		except Exception as e:
			logging.error(f"Error analyzing batch: {batch}, {e}")

	return trade_opportunities


from functools import lru_cache

from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator

@lru_cache(maxsize=128)
def calculate_cached_indicators(data):
	"""
	Calculate technical indicators and cache the results.

	Args:
		data (pd.DataFrame): Market data with 'close', 'high', and 'low' columns.

	Returns:
		pd.DataFrame: DataFrame with added indicators.
	"""
	try:
		# Ensure required columns exist
		if not {'close', 'high', 'low'}.issubset(data.columns):
			raise ValueError("Missing required columns in input data.")

		# Calculate indicators
		data['EMA_20'] = EMAIndicator(data['close'], window=20).ema_indicator()
		data['EMA_50'] = EMAIndicator(data['close'], window=50).ema_indicator()
		data['RSI'] = RSIIndicator(data['close'], window=14).rsi()

		# Drop NaN values resulting from rolling calculations
		data = data.dropna()

		return data
	except Exception as e:
		logging.error(f"Error in calculate_cached_indicators: {e}")
		return pd.DataFrame()  # Return an empty DataFrame on failure



def calculate_optimal_interval(data):
	"""
	Calculate optimal interval based on market volatility.

	Args:
		data (pd.DataFrame): Historical data.

	Returns:
		int: Optimal interval in seconds.
	"""
	atr = data['ATR'].iloc[-1]  # Assume ATR is calculated as part of indicators
	return max(10, int(60 / (atr / data['close'].iloc[-1])))  # Adjust interval dynamically


def controlled_logging(message, level='info'):
	"""
	Log messages conditionally based on the logging level.

	Args:
		message (str): Message to log.
		level (str): Logging level ('info', 'warning', 'error').
	"""
	if level == 'info':
		logging.info(message)
	elif level == 'warning':
		logging.warning(message)
	elif level == 'error':
		logging.error(message)


#Prepare Training Data
async def prepare_training_data(symbols):
	"""
	Prepare combined training data for all symbols.

	Args:
		symbols (list): List of trading symbols.

	Returns:
		pd.DataFrame: Combined DataFrame of historical data for all symbols.
	"""
	logging.info("Fetching historical data for training...")
	symbol_data = await fetch_all_symbols_data_for_futures(symbols)

	# Combine valid DataFrames into one
	valid_data = [df for df in symbol_data.values() if df is not None]
	if not valid_data:
		logging.warning("No valid data available for training.")
		return pd.DataFrame()  # Return empty DataFrame if no data

	combined_data = pd.concat(valid_data, ignore_index=True)
	return combined_data



"""
demo_api_key = "4f0c5b7c0227d06b6071b9f9954f1f692e073a3050354b5a0b4722623ea18b59"
demo_api_secret = "93d48d4cb106a5b2dd74db56e6388b961a02daee63afe1be6ea3649abeed30cb"

binance = ccxt.binance({
	'apiKey': demo_api_key,
	'secret': demo_api_secret,
	'options': {
		'defaultType': 'future',
		'test': True  # Ensure test mode is explicitly set
	},
	'urls': {
		'api': {
			'public': 'https://testnet.binancefuture.com/fapi/v1',
			'private': 'https://testnet.binancefuture.com/fapi/v1'
		}
	}
})
"""

# Environment variables for sensitive data
position_data = {}
api_key = "pNsQEHYQ4oNqUasdasdTp1niCl0rTUbzOghTDh4byl438O7HTtlNezES93nJlJf5cfT1ug"
api_secret = "3PSiaHxR0awasdasdsjlHGfqovw4346PyN1E9pxds658yu6zZRCKnPTwmox9Xs8nMOXYT04bf"
openai_api_key = os.getenv('sk-proj-LnceXqYdtgCddL-XkxXxtQxISdcZ-VXpUsjyxsad30Abl1mrsdAok_sdbqC_OCbS35Cs123xckxXUpuSWxcR1k5rkP9E-9T3BlbkFasJ2dasdABj-mAL9VhEM8AhjAaASRZfy7F7XDkDSkCKVvASSWbo_q9YZ5m3rfASi6v-pOA1GkxPfPJePInDs92C3wAA')

# Binance API Configuration
binance = ccxt.binance({
	'apiKey': api_key,
	'secret': api_secret,
	'options': {'defaultType': 'future'}
})


# Google Cloud Configuration
GCP_BUCKET_NAME = "tiderungen"
storage_client = storage.Client()

# OpenAI API Key
openai.api_key = "sk-proj-LnceXqYdtgCddL-XkxXxtQxISdcZ-VXpUsjyxsad30Abl1mrsdAok_sdbqC_OCbS35Cs123xckxXUpuSWxcR1k5rkP9E-9T3BlbkFasJ2dasdABj-mAL9VhEM8AhjAaASRZfy7F7XDkDSkCKVvASSWbo_q9YZ5m3rfASi6v-pOA1GkxPfPJePInDs92C3wAA"

# Global Variables
ml_model = None
scheduler = AsyncIOScheduler()

import asyncio
import logging
import os
import signal
import sys
import io
# For Windows compatibility, use a manual shutdown flag
shutdown_event = asyncio.Event()




def graceful_shutdown(*args):
	logging.info("Shutting down gracefully...")
	shutdown_event.set()

if sys.platform.startswith("win"):
	import win32api
	win32api.SetConsoleCtrlHandler(lambda sig: graceful_shutdown(), True)
else:
	loop = asyncio.get_event_loop()
	loop.add_signal_handler(signal.SIGINT, graceful_shutdown)
	loop.add_signal_handler(signal.SIGTERM, graceful_shutdown)


"""

async def test_binance_api():
	try:
		balance = await binance.fetch_balance()
		print(balance)
	except Exception as e:
		print(f"API Error: {e}")

asyncio.run(test_binance_api())
"""
import websockets
import json

# Market Monitoring
async def monitor_market(symbol):
	url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@ticker"
	async with websockets.connect(url) as websocket:
		while True:
			data = json.loads(await websocket.recv())
			spread = float(data['askPrice']) - float(data['bidPrice'])
			volatility = calculate_market_volatility(symbol)  # Call volatility function
			logging.info(f"{symbol} | Spread: {spread:.2f} | Volatility: {volatility:.2f}%")

			if volatility > 5:
				logging.warning(f"High volatility detected for {symbol}. Adjusting strategy dynamically.")



# Multi Timeframe Analysis
async def fetch_data_for_timeframes(symbol, timeframes=['15m', '1h', '1d']):
	"""Fetch data for multiple timeframes and calculate indicators."""
	results = {}
	for tf in timeframes:
		df = await fetch_and_save_historical_data(symbol, timeframe=tf, limit=500)
		if not df.empty:
			results[tf] = calculate_indicators(df)
	return results

def check_timeframe_alignment(dataframes):
	"""Check alignment of indicators across timeframes."""
	try:
		long_signals = 0
		short_signals = 0

		for tf, df in dataframes.items():
			if 'EMA_20' not in df or 'EMA_50' not in df:
				logging.warning(f"Missing EMA indicators in {tf} timeframe. Skipping alignment check for this timeframe.")
				continue

			if df['EMA_20'].iloc[-1] > df['EMA_50'].iloc[-1]:
				long_signals += 1
			elif df['EMA_20'].iloc[-1] < df['EMA_50'].iloc[-1]:
				short_signals += 1

		if long_signals > short_signals:
			return 'long'
		elif short_signals > long_signals:
			return 'short'
		else:
			return 'hold'
	except Exception as e:
		logging.error(f"Error in timeframe alignment check: {e}")
		return 'hold'



# Machine Learning Enhancements
def feature_importance_analysis(model, features):
	"""
	Analyze and log feature importance.

	Args:
		model: Trained machine learning model.
		features (pd.DataFrame): Feature DataFrame.

	Returns:
		dict: Feature importance mapping.
	"""
	try:
		importances = model.feature_importances_
		feature_importance = {col: imp for col, imp in zip(features.columns, importances)}
		logging.info("Feature Importances:")
		for feature, importance in feature_importance.items():
			logging.info(f"Feature: {feature}, Importance: {importance:.4f}")
		return feature_importance
	except Exception as e:
		logging.error(f"Error analyzing feature importance: {e}")
		return {}


def detect_data_drift(old_data, new_data, threshold=0.1):
	"""Detect data drift using statistical comparison."""
	drifts = {}
	for column in old_data.columns:
		old_mean, new_mean = old_data[column].mean(), new_data[column].mean()
		drift = abs(old_mean - new_mean) / max(abs(old_mean), abs(new_mean))
		drifts[column] = drift > threshold
	return drifts

def train_ensemble_model(data):
	"""Train an ensemble model."""
	from sklearn.ensemble import VotingClassifier
	from xgboost import XGBClassifier
	from sklearn.ensemble import RandomForestClassifier

	features = data.drop(['timestamp', 'close'], axis=1)
	target = np.where(data['close'].shift(-1) > data['close'], 1, 0)

	models = [
		('lgbm', LGBMClassifier()),
		('xgb', XGBClassifier()),
		('rf', RandomForestClassifier())
	]
	ensemble = VotingClassifier(estimators=models, voting='soft')
	ensemble.fit(features, target)
	return ensemble

#Performance Metrics and Reporting
import numpy as np

def calculate_performance_metrics(trades, initial_balance):
    """
    Calculate performance metrics from trades.

    Args:
        trades (list): List of executed trades with 'pnl' values.
        initial_balance (float): Starting balance for the backtest.

    Returns:
        dict: Dictionary of calculated performance metrics.
    """
    try:
        pnl = [trade.get('pnl', 0) for trade in trades]
        net_profit = sum(pnl)
        win_rate = len([p for p in pnl if p > 0]) / len(pnl) * 100 if pnl else 0
        max_drawdown = min(pnl) / initial_balance if pnl else 0
        sharpe_ratio = np.mean(pnl) / np.std(pnl) if len(pnl) > 1 else 0

        return {
            "net_profit": net_profit,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
        }
    except Exception as e:
        logging.error(f"Error calculating performance metrics: {e}")
        return {
            "net_profit": 0,
            "win_rate": 0,
            "max_drawdown": 0,
            "sharpe_ratio": 0,
        }


def calculate_max_drawdown(trades, initial_balance):
	"""
	Calculate the maximum drawdown from trade results.

	Args:
		trades (list): List of trades with 'pnl'.
		initial_balance (float): Starting balance.

	Returns:
		float: Maximum drawdown percentage.
	"""
	balance = initial_balance
	equity_curve = [balance]

	for trade in trades:
		balance += trade['pnl']
		equity_curve.append(balance)

	peak = max(equity_curve)
	trough = min(equity_curve[equity_curve.index(peak):])
	drawdown = (peak - trough) / peak * 100 if peak > 0 else 0

	return drawdown


def generate_report(metrics):
	"""Generate a performance report."""
	report = "\n".join([f"{key}: {value:.2f}" for key, value in metrics.items()])
	with open("performance_report.txt", "w") as f:
		f.write(report)
	logging.info("Performance report saved.")


#Backtesting Enhancements
def simulate_trade_execution(price, amount, fee=0.001, slippage=0.01):
	"""Simulate trade execution with slippage and fees."""
	executed_price = price * (1 + slippage)
	fees = executed_price * amount * fee
	return executed_price * amount - fees


#Enhanced Backtest Strategy
async def enhanced_backtest_strategy(df, strategy_fn, initial_balance, fee=0.001):
    """
    Superior backtesting strategy with dynamic trade amount, leverage, PnL tracking, and error handling.

    Args:
        df (pd.DataFrame): Historical market data.
        strategy_fn (callable): Trading strategy function.
        initial_balance (float): Starting balance for backtesting.
        fee (float): Trading fee percentage.

    Returns:
        dict: Backtesting results including total PnL and trade details.
    """
    balance = initial_balance
    position = 0
    entry_price = 0
    trades = []

    logging.info("Starting enhanced backtest...")

    # Check if the strategy function is callable
    if not callable(strategy_fn):
        logging.error("The strategy function is not callable. Please verify its implementation.")
        return {"total_pnl": 0, "trades": [], "error": "Invalid strategy function"}

    try:
        for i in range(len(df)):
            # Apply the strategy to generate a signal
            signal = strategy_fn(df.iloc[:i + 1])

            # Log current row, signal, position, and balance for debugging
            logging.debug(f"Row {i}: {df.iloc[i].to_dict()}")
            logging.debug(f"Signal: {signal}, Position: {position}, Balance: {balance}")

            # Open a position on a "buy" signal
            if signal == "buy" and position == 0:
                try:
                    # Dynamic trade amount and leverage adjustment
                    trade_amount, leverage = await adjust_leverage_and_amount(df['symbol'].iloc[0], balance)
                    if trade_amount == 0:
                        logging.warning("Trade amount is 0. Skipping trade.")
                        continue

                    entry_price = df['close'].iloc[i]
                    position = trade_amount / entry_price  # Calculate position size based on trade amount
                    balance -= position * entry_price * (1 + fee)  # Deduct trade cost and fee
                    trades.append({
                        "action": "buy",
                        "price": entry_price,
                        "amount": position,
                        "leverage": leverage,
                        "pnl": 0  # PnL is 0 for buy trades
                    })
                    logging.info(f"Buy executed at {entry_price} with position size: {position} and leverage: {leverage}.")
                except Exception as e:
                    logging.error(f"Error during buy execution: {e}")

            # Close the position on a "sell" signal
            elif signal == "sell" and position > 0:
                try:
                    exit_price = df['close'].iloc[i]
                    pnl = position * (exit_price - entry_price) - (fee * position * (entry_price + exit_price))
                    balance += position * exit_price * (1 - fee)  # Add realized PnL to balance
                    trades.append({
                        "action": "sell",
                        "price": exit_price,
                        "amount": position,
                        "pnl": pnl
                    })
                    logging.info(f"Sell executed at {exit_price}. PnL: {pnl}.")
                    position = 0  # Reset position
                except Exception as e:
                    logging.error(f"Error during sell execution: {e}")

        # Finalize any open positions
        if position > 0:
            try:
                final_price = df['close'].iloc[-1]
                pnl = position * (final_price - entry_price) - (fee * position * (entry_price + final_price))
                balance += position * final_price * (1 - fee)  # Add final PnL to balance
                trades.append({
                    "action": "sell",
                    "price": final_price,
                    "amount": position,
                    "pnl": pnl
                })
                logging.info(f"Final sell executed at {final_price}. PnL: {pnl}.")
                position = 0  # Reset position
            except Exception as e:
                logging.error(f"Error finalizing position: {e}")

        # Calculate total PnL
        total_pnl = sum(trade.get("pnl", 0) for trade in trades)
        logging.info(f"Total PnL: {total_pnl}")

        return {"total_pnl": total_pnl, "trades": trades}

    except Exception as e:
        logging.error(f"Unexpected error in backtest: {e}")
        return {"total_pnl": 0, "trades": [], "error": str(e)}







async def fetch_sentiment_scores(symbols):
	"""
	Fetch sentiment scores for a list of symbols.

	Args:
		symbols (list): List of symbols to fetch sentiment scores for.

	Returns:
		dict: Sentiment scores for each symbol.
	"""
	sentiment_scores = {}
	try:
		for symbol in symbols:
			sentiment_scores[symbol] = await fetch_sentiment_data_for_symbol(symbol)  # Example function
		logging.info("Fetched sentiment scores for symbols.")
	except Exception as e:
		logging.error(f"Error fetching sentiment scores: {e}")

	return sentiment_scores


async def mock_sentiment_api_call(symbol):
	"""
	Mock function simulating sentiment API call.
	Args:
		symbol (str): Symbol for which sentiment score is requested.
	Returns:
		float: Simulated sentiment score between -1 (negative) and 1 (positive).
	"""
	import random
	await asyncio.sleep(0.1)  # Simulating network delay
	return random.uniform(-1, 1)



def integrate_sentiment_into_decisions(trade_signals, sentiment_scores):
	"""
	Modify trade signals based on sentiment scores.

	Args:
		trade_signals (dict): Trade signals generated by the strategy.
		sentiment_scores (dict): Sentiment scores for each symbol.

	Returns:
		dict: Updated trade signals influenced by sentiment scores.
	"""
	updated_signals = {}
	try:
		for symbol, signal in trade_signals.items():
			sentiment = sentiment_scores.get(symbol, 0)
			if sentiment > 0.5 and signal == "sell":
				updated_signals[symbol] = "hold"
			elif sentiment < -0.5 and signal == "buy":
				updated_signals[symbol] = "hold"
			else:
				updated_signals[symbol] = signal

		logging.info("Integrated sentiment scores into trade decisions.")
	except Exception as e:
		logging.error(f"Error integrating sentiment into decisions: {e}")

	return updated_signals



async def integrate_backtest_results(backtest_results):
	"""
	Integrate backtest results into the live trading strategy.
	Args:
		backtest_results (pd.DataFrame): DataFrame of backtest results.
	Returns:
		dict: Refined strategy configuration.
	"""
	refined_strategy = {}
	try:
		# Extract meaningful patterns from backtest results
		for _, row in backtest_results.iterrows():
			symbol = row['symbol']
			if row['sharpe_ratio'] > 1.5:
				refined_strategy[symbol] = "aggressive"
			elif row['sharpe_ratio'] > 0.5:
				refined_strategy[symbol] = "moderate"
			else:
				refined_strategy[symbol] = "conservative"
		logging.info("Backtest results successfully integrated into strategy.")
	except Exception as e:
		logging.error(f"Error integrating backtest results: {e}")
	return refined_strategy


def performance_drops_below_threshold():
	"""
	Check if performance metrics drop below predefined thresholds.
	Returns:
		bool: True if performance has dropped, False otherwise.
	"""
	try:
		threshold_net_profit = -100
		threshold_sharpe_ratio = 1.0
		threshold_win_rate = 0.5

		current_net_profit = pnl_metric.collect()[0].samples[0].value
		current_sharpe_ratio = sharpe_ratio_metric.collect()[0].samples[0].value
		current_win_rate = win_rate_metric.collect()[0].samples[0].value

		if (current_net_profit < threshold_net_profit or
				current_sharpe_ratio < threshold_sharpe_ratio or
				current_win_rate < threshold_win_rate):
			logging.warning("Performance below thresholds.")
			return True
		return False
	except Exception as e:
		logging.error(f"Error checking performance thresholds: {e}")
		return False




def collect_backtest_metrics() -> dict:
	"""
	Collects backtest performance metrics.

	Returns:
		dict: Dictionary of backtest performance metrics.
	"""
	return {
		"average_pnl": backtest_metrics.get("average_pnl", 0),
		"win_rate": backtest_metrics.get("win_rate", 0),
		"max_drawdown": backtest_metrics.get("max_drawdown", 0),
	}



def collect_sentiment_metrics() -> dict:
	"""
	Collects sentiment metrics for trading decisions.

	Returns:
		dict: Dictionary of sentiment metrics.
	"""
	return {
		"sentiment_score": sentiment_metric.get(),
	}


def collect_live_metrics() -> dict:
	"""
	Collects live performance metrics such as Sharpe ratio and win rate.

	Returns:
		dict: Dictionary of live performance metrics.
	"""
	return {
		"sharpe_ratio": sharpe_ratio_metric.get(),
		"win_rate": win_rate_metric.get(),
	}



def update_strategy(new_strategy):
	"""
	Update the current strategy dynamically.
	Args:
		new_strategy (dict): Refined strategy details.
	"""
	global strategy_fn_with_regression
	strategy_fn_with_regression = new_strategy
	logging.info("Updated strategy with GPT-refined version.")


import openai

async def gpt_model_refinement(metrics_data: dict) -> dict:
	"""
	Refines trading strategies using GPT-based analysis of performance, sentiment, and backtest metrics.

	Args:
		metrics_data (dict): Dictionary containing live, sentiment, and backtest metrics.

	Returns:
		dict: Refined strategy parameters or suggestions.
	"""
	try:
		# Prepare prompt for GPT model
		prompt = (
			"Based on the following performance metrics, sentiment analysis, and backtest results, "
			"suggest refinements for the trading strategy:\n\n"
			f"Performance Metrics:\n{metrics_data.get('performance', 'No data')}\n\n"
			f"Sentiment Metrics:\n{metrics_data.get('sentiment', 'No data')}\n\n"
			f"Backtest Metrics:\n{metrics_data.get('backtest', 'No data')}\n\n"
			"Provide specific adjustments to parameters, risk management, or indicators."
		)

		# GPT-4 API call
		response = openai.Completion.create(
			engine="gpt-4",
			prompt=prompt,
			max_tokens=500,
			temperature=0.7,
		)
		strategy_refinement = response.choices[0].text.strip()
		logging.info(f"GPT Refinement Suggestions: {strategy_refinement}")

		# Process GPT suggestions into usable strategy parameters
		return {"strategy_adjustments": strategy_refinement}
	except Exception as e:
		logging.error(f"Error during GPT model refinement: {e}")
		return {}


async def refine_strategy_with_gpt():
	"""
	Collects live, sentiment, and backtest metrics to refine the trading strategy dynamically using GPT.

	Returns:
		dict: Refined strategy adjustments.
	"""
	try:
		# Collect metrics
		performance_data = collect_live_metrics()
		sentiment_data = collect_sentiment_metrics()
		backtest_data = collect_backtest_metrics()

		# Combine metrics into a single data structure
		metrics_data = {
			"performance": performance_data,
			"sentiment": sentiment_data,
			"backtest": backtest_data,
		}

		# Call GPT-based refinement function
		refined_strategy = await gpt_model_refinement(metrics_data)

		return refined_strategy
	except Exception as e:
		logging.error(f"Error during strategy refinement with GPT: {e}")
		return {}



async def live_trading_with_refinements(symbols, balance):
	"""
	Live trading loop with sentiment integration and GPT-based refinements.
	"""
	while True:
		try:
			logging.info("Fetching sentiment scores...")
			sentiment_scores = fetch_sentiment_scores(symbols)
			
			logging.info("Analyzing market opportunities...")
			trade_opportunities = await analyze_in_batches(symbols, strategy_fn_with_regression, batch_size=5)

			for trade in trade_opportunities:
				symbol = trade['symbol']
				position = trade['position']
				sentiment = sentiment_scores.get(symbol, 0)

				# Adjust decisions based on sentiment
				adjusted_decision = integrate_sentiment_into_decisions(trade, sentiment)
				logging.info(f"Adjusted decision for {symbol}: {adjusted_decision}")

				# Execute trade
				await execute_trade(adjusted_decision, balance)

			# Check performance and trigger refinements
			if performance_drops_below_threshold():
				logging.info("Performance dropped. Triggering GPT-based refinements.")
				refine_strategy_with_gpt()

			await asyncio.sleep(60)  # Pause before next iteration
		except Exception as e:
			logging.error(f"Error in live trading loop: {e}")
			break







def verify_time_alignment(dataframes):
	"""
	Verify the alignment of timestamps across multiple timeframes.

	Args:
		dataframes (dict): Dictionary of timeframes to DataFrames.

	Returns:
		bool: True if all timeframes are aligned, False otherwise.
	"""
	try:
		timestamps = [df['timestamp'] for df in dataframes.values()]
		aligned = all(t.equals(timestamps[0]) for t in timestamps)
		if not aligned:
			logging.warning("Timestamps across timeframes are not aligned.")
		return aligned
	except Exception as e:
		logging.error(f"Error verifying time alignment: {e}")
		return False
	
async def fetch_and_verify_multi_timeframe_data(symbol, timeframes):
	"""
	Fetch and verify data across multiple timeframes.

	Args:
		symbol (str): Trading symbol.
		timeframes (list): List of timeframes.

	Returns:
		dict: Dictionary of timeframes to DataFrames if aligned, otherwise None.
	"""
	dataframes = {}
	for timeframe in timeframes:
		df = await fetch_and_save_historical_data(symbol, timeframe)
		if validate_dataset(df):
			dataframes[timeframe] = df

	if verify_time_alignment(dataframes):
		return dataframes
	else:
		logging.error(f"Time alignment failed for {symbol} across timeframes: {timeframes}")
		return None



#Dynamic Hyperparameter Tuning
def optimize_lgbm_hyperparameters(features, target):
	"""Optimize LightGBM hyperparameters dynamically."""
	try:
		param_grid = {
			'max_depth': [3, 5, 7],
			'learning_rate': [0.01, 0.1, 0.2],
			'n_estimators': [50, 100, 200],
		}
		model = LGBMClassifier()
		grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3)
		grid_search.fit(features, target)
		best_params = grid_search.best_params_
		logging.info(f"Optimized Parameters: {best_params}")
		return best_params
	except Exception as e:
		logging.error(f"Error in hyperparameter optimization: {e}")
		return None

def calculate_market_volatility(df, period=14):
	"""
	Calculate market volatility using Average True Range (ATR).
	Args:
		df (pd.DataFrame): Historical market data with 'high', 'low', and 'close' columns.
		period (int): Lookback period for ATR calculation.
	Returns:
		float: Latest ATR value as a measure of volatility.
	"""
	try:
		atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=period).average_true_range()
		return atr.iloc[-1]  # Latest volatility value
	except Exception as e:
		logging.error(f"Error calculating market volatility: {e}")
		return None



def calculate_pnl(trades):
	"""
	Calculate the cumulative Profit and Loss (PnL) from executed trades.
	Args:
		trades (list): List of trade dictionaries, each containing 'price', 'amount', and 'action' keys.
	Returns:
		float: Cumulative PnL value.
	"""
	try:
		pnl = 0
		for trade in trades:
			if trade['action'] == 'sell':
				pnl += trade['price'] * trade['amount'] - trade.get('entry_price', 0) * trade['amount']
		return pnl
	except Exception as e:
		logging.error(f"Error calculating PnL: {e}")
		return 0




# Save To Google Cloud
import aiohttp
import json
# Load Data from Google Cloud Storage
from google.cloud.storage import Client
import aiohttp
import pandas as pd
import io
import logging
async def async_save_to_google_cloud(data, filename):
	try:
		bucket = storage_client.bucket(GCP_BUCKET_NAME)
		blob = bucket.blob(f"{filename}.gz")

		buffer = io.BytesIO()
		if isinstance(data, pd.DataFrame):
			data.to_csv(buffer, index=False, compression="gzip")
		else:
			json.dump(data, buffer)
		buffer.seek(0)

		logging.info(f"Uploading {filename} to Google Cloud Storage...")
		blob.upload_from_file(buffer, content_type="application/gzip")
		logging.info(f"Successfully uploaded {filename}.gz to Google Cloud.")
	except Exception as e:
		logging.error(f"Error saving {filename} to Google Cloud: {e}")

async def async_load_from_google_cloud(filename):
	try:
		bucket = storage_client.bucket(GCP_BUCKET_NAME)
		blob = bucket.blob(filename)
		if not blob.exists():
			logging.warning(f"{filename} not found in Google Cloud.")
			return None

		data = blob.download_as_string()
		df = pd.read_csv(io.StringIO(data))
		logging.info(f"Successfully loaded {filename} from Google Cloud.")
		return df
	except Exception as e:
		logging.error(f"Error loading {filename} from Google Cloud: {e}")
		return None








	
def save_model_to_cloud(model, filename="lightgbm_model.pkl"):
	"""Save LightGBM model to Google Cloud."""
	try:
		with open(filename, 'wb') as f:
			pickle.dump(model, f)
		bucket = storage_client.bucket(GCP_BUCKET_NAME)
		blob = bucket.blob(filename)
		blob.upload_from_filename(filename)
		logging.info(f"Model saved to Google Cloud: {filename}")
	except Exception as e:
		logging.error(f"Error saving model to Google Cloud: {e}")


def load_model_from_cloud(filename="lightgbm_model.pkl"):
	"""Load LightGBM model from Google Cloud."""
	try:
		local_path = f"/tmp/{filename}"
		bucket = storage_client.bucket(GCP_BUCKET_NAME)
		blob = bucket.blob(filename)
		blob.download_to_filename(local_path)

		with open(local_path, 'rb') as f:
			model = pickle.load(f)
		logging.info("Model successfully loaded from Google Cloud.")
		return model
	except Exception as e:
		logging.error(f"Error loading model from Google Cloud: {e}")
		return None




async def save_metrics_to_google_cloud(metrics, filename="model_metrics.json"):
	"""Save model metrics to Google Cloud."""
	try:
		await async_save_to_google_cloud(metrics, filename)
		logging.info(f"Saved metrics to Google Cloud: {filename}")
	except Exception as e:
		logging.error(f"Error saving metrics to Google Cloud: {e}")





# Start Prometheus metrics server

from prometheus_client import REGISTRY, start_http_server, Gauge
import logging

# Prometheus metrics variables
latency_metric = None
api_error_metric = None
processing_time_metric = None
pnl_metric = None
trade_count_metric = None

def initialize_prometheus(port=8000):
    """
    Initialize Prometheus metrics and start the HTTP server without unregistering existing data.

    Args:
        port (int): Port to expose Prometheus metrics.
    """
    global latency_metric, api_error_metric, processing_time_metric, pnl_metric, trade_count_metric

    def get_or_create_metric(metric_name, metric_description):
        """
        Reuse an existing metric or create a new one if it doesn't exist.
        """
        if metric_name in REGISTRY._names_to_collectors:
            logging.info(f"Metric '{metric_name}' already registered. Reusing existing collector.")
            return REGISTRY._names_to_collectors[metric_name]
        else:
            logging.info(f"Registering new metric: {metric_name}")
            return Gauge(metric_name, metric_description)

    try:
        # Define or reuse metrics
        latency_metric = get_or_create_metric('api_call_latency', 'Latency of API calls in seconds')
        api_error_metric = get_or_create_metric('api_errors', 'Number of API errors encountered')
        processing_time_metric = get_or_create_metric('processing_time', 'Average processing time per task (in seconds)')
        pnl_metric = get_or_create_metric('trading_pnl', 'Profit and Loss')
        trade_count_metric = get_or_create_metric('trading_trades', 'Number of trades executed')

        # Start Prometheus HTTP server only if it's not already started
        try:
            start_http_server(port)
            logging.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logging.warning(f"Prometheus server might already be running: {e}")
    except Exception as e:
        logging.error(f"Error initializing Prometheus metrics: {e}")



















# GPT-Based Strategy Refinement
import json
import asyncio

import json
import asyncio

async def refined_strategy_with_gpt_retry(strategy, max_retries=3, delay=2):
	query = "Optimize the given strategy for better performance."
	for attempt in range(max_retries):
		try:
			response = await openai.ChatCompletion.acreate(
				model="gpt-4",
				messages=[
					{"role": "system", "content": "You are a trading assistant."},
					{"role": "user", "content": f"Context: {json.dumps(strategy)}\nQuery: {query}"}
				]
			)
			refined_strategy = json.loads(response['choices'][0]['message']['content'])
			if isinstance(refined_strategy, dict):  # Validate response
				return refined_strategy
		except Exception as e:
			print(f"GPT refinement failed (Attempt {attempt + 1}): {e}")
			await asyncio.sleep(delay)
	return strategy




def monte_carlo_simulation(strategy_fn, data, num_simulations=100):
	results = []
	for _ in range(num_simulations):
		perturbed_data = data.copy()
		perturbed_data['close'] *= (1 + np.random.normal(0, 0.01, len(data)))
		results.append(strategy_fn(perturbed_data))
	return results


def generate_daily_summary(metrics):
	summary = f"""
	Daily Trading Summary:
	----------------------
	Total PnL: {metrics['pnl']:.2f}
	Number of Trades: {metrics['trade_count']}
	"""
	with open("daily_summary.txt", "w") as file:
		file.write(summary)
	print("Daily summary saved.")




from prometheus_client import Gauge, CollectorRegistry





def update_metrics(pnl, trades):
	pnl_metric.set(pnl)
	trade_count_metric.inc(trades)

def start_prometheus_server(port=8000):
	prom.start_http_server(port)
	logging.info(f"Prometheus metrics server started on port {port}")


from skopt import BayesSearchCV

def optimize_model_with_bayesian(features, target):
	"""Optimize LightGBM using Bayesian Optimization."""
	try:
		param_grid = {
			'max_depth': (3, 10),
			'learning_rate': (0.01, 0.3, 'log-uniform'),
			'n_estimators': (50, 200),
			'num_leaves': (20, 100)
		}
		model = LGBMClassifier()
		search = BayesSearchCV(estimator=model, search_spaces=param_grid, scoring='accuracy', cv=3, n_iter=50)
		search.fit(features, target)
		logging.info(f"Optimized Parameters: {search.best_params_}")
		return search.best_estimator_
	except Exception as e:
		logging.error(f"Error in model optimization: {e}")
		return None

async def retrain_model_if_needed(symbols, performance_threshold=0.7):
	"""Retrain the model if real-time performance drops below the threshold."""
	try:
		live_metrics = await async_load_from_google_cloud("live_metrics.json", is_dataframe=False)
		if not live_metrics:
			logging.info("No live metrics available. Skipping retraining.")
			return

		current_performance = live_metrics.get('accuracy', 1.0)
		if current_performance < performance_threshold:
			logging.warning(f"Performance dropped to {current_performance}. Retraining model.")
			combined_data = pd.concat([await fetch_historical_data_sync(symbol) for symbol in symbols], ignore_index=True)
			model = train_model(combined_data)
			save_model_to_cloud(model)
			logging.info("Model retrained and saved.")
	except Exception as e:
		logging.error(f"Error during retraining: {e}")

def calculate_dynamic_position_size(balance, atr, risk_percentage=0.01):
	"""Calculate position size dynamically based on account balance and ATR."""
	try:
		if atr <= 0:
			raise ValueError("ATR must be positive.")
		position_size = (balance * risk_percentage) / atr
		return round(position_size, 3)
	except Exception as e:
		logging.error(f"Error calculating position size: {e}")
		return 0

def calculate_adaptive_stop_loss(df, position):
	"""Calculate stop-loss dynamically based on volatility."""
	try:
		atr = df['ATR'].iloc[-1]
		close_price = df['close'].iloc[-1]
		if position == 'long':
			stop_loss = close_price - (2 * atr)
		else:
			stop_loss = close_price + (2 * atr)
		return stop_loss
	except Exception as e:
		logging.error(f"Error calculating adaptive stop loss: {e}")
		return None



# ChatGPT Feedback Loop
async def feedback_loop_gpt(context, query):
	try:
		response = await openai.ChatCompletion.acreate(
			model="gpt-4",
			messages=[
				{"role": "system", "content": "You are an AI trading assistant."},
				{"role": "user", "content": f"Context: {context}\nQuery: {query}"}
			],
			max_tokens=500,
			temperature=0.7
		)
		logging.info(f"GPT API call succeeded with response: {response['choices'][0]['message']['content']}")
		return response['choices'][0]['message']['content']
	except Exception as e:
		logging.error(f"GPT API call failed: {e}")
		return None






	
# Get GPT Suggestions for Strategy
async def get_gpt_suggestions(strategy, retries=3, delay=2):
	"""Refine trading strategy using GPT feedback with retry mechanism."""
	context = f"Current strategy:\n{json.dumps(strategy, indent=2)}"
	query = "What adjustments would you recommend for better performance?"
	
	for attempt in range(retries):
		try:
			response = await feedback_loop_gpt(context, query)
			refined_strategy = json.loads(response)
			logging.info("GPT suggestions applied successfully.")
			return refined_strategy
		except openai.error.RateLimitError:
			logging.warning(f"Rate limit hit. Retrying in {delay}s...")
			await asyncio.sleep(delay)
			delay *= 2  # Exponential backoff
		except json.JSONDecodeError:
			logging.warning("GPT response is not valid JSON.")
		except Exception as e:
			logging.warning(f"GPT suggestion failed (Attempt {attempt + 1}): {e}")
			await asyncio.sleep(delay * (2 ** attempt))
	logging.error("Failed to get GPT suggestions after retries.")
	return strategy



def fetch_data_in_chunks(symbols, chunk_size=5):
	for i in range(0, len(symbols), chunk_size):
		yield symbols[i:i + chunk_size]

async def process_data_chunks(symbols):
	for chunk in fetch_data_in_chunks(symbols):
		for symbol in chunk:
			df = await fetch_and_save_historical_data(symbol)
			# Process individual symbol data

async def process_symbol(symbol):
	try:
		df = await fetch_and_save_historical_data(symbol)
		# Further process df
	except Exception as e:
		logging.error(f"Error processing symbol {symbol}: {e}")

async def process_all_symbols(symbols):
	await asyncio.gather(*(process_symbol(symbol) for symbol in symbols))

def is_data_stale(filename, refresh_interval=24):
	try:
		bucket = storage_client.bucket(GCP_BUCKET_NAME)
		blob = bucket.blob(filename)
		last_updated = blob.updated
		return (datetime.utcnow() - last_updated).total_seconds() > refresh_interval * 3600
	except Exception as e:
		logging.error(f"Error checking data freshness: {e}")
		return True


async def should_retrain(data, last_retrained, retrain_interval_hours=24):
	current_time = datetime.utcnow()
	if (current_time - last_retrained).total_seconds() / 3600 < retrain_interval_hours:
		logging.info("Retraining skipped: Interval not met.")
		return False

	# Check data changes
	if not has_data_changed(data):
		logging.info("Retraining skipped: No significant data changes.")
		return False

	return True



async def retrain_if_needed(symbols):
	"""
	Trigger retraining if necessary based on the retraining condition.
	"""
	if should_retrain():
		logging.info("Retraining triggered.")
		await retrain_model(symbols)
	else:
		logging.info("Skipping retraining: Conditions not met.")



import requests

async def fetch_news_headlines(symbol):
	"""Fetch news headlines for the given symbol from NewsAPI."""
	try:
		# Fetch your API key from the environment or hard-code it for testing
		api_key = os.getenv("NEWS_API_KEY", "65270c19asdasdf6123994acdbdd8dc8bb6avcd463dec0")  # Replace with your API key
		if not api_key:
			raise ValueError("NewsAPI key is not set. Please set the NEWS_API_KEY environment variable.")
		
		# Base URL for NewsAPI
		base_url = "https://newsapi.org/v2/everything"
		
		# Format the query for the symbol (you may adjust keywords as necessary)
		query = f"{symbol} stock OR {symbol} trading OR {symbol} market"

		# Make the request to NewsAPI
		params = {
			"q": query,
			"apiKey": "65270c19f69232394acdb8dsddc8b64asadd463dddec0",
			"sortBy": "publishedAt",  # Sort news by the most recent
			"language": "en"  # Fetch only English news
		}
		response = requests.get(base_url, params=params)

		# Parse response
		if response.status_code == 200:
			data = response.json()
			articles = data.get("articles", [])
			headlines = [article["title"] for article in articles]
			logging.info(f"Fetched {len(headlines)} headlines for {symbol}.")
			return headlines
		else:
			logging.error(f"Failed to fetch news: {response.status_code}, {response.text}")
			return []
	except Exception as e:
		logging.error(f"Error fetching news for {symbol}: {e}")
		return []




async def analyze_market_sentiment(symbol):
	"""Fetch and analyze market sentiment using GPT."""
	try:
		news_headlines = await fetch_news_headlines(symbol)
		if not news_headlines:
			logging.warning("No news headlines found. Defaulting sentiment to neutral.")
			return 0.5  # Neutral sentiment

		prompt = f"Analyze the sentiment of these headlines related to {symbol}:\n{news_headlines}\n" \
				 f"Provide a sentiment score between 0 (very negative) and 1 (very positive)."

		sentiment = await feedback_loop_gpt("Market Sentiment Analysis", prompt)
		sentiment_score = float(sentiment.strip())

		if 0 <= sentiment_score <= 1:
			return sentiment_score
		else:
			logging.warning(f"Invalid sentiment score: {sentiment_score}. Defaulting to neutral.")
			return 0.5
	except (ValueError, TypeError):
		logging.warning("Invalid sentiment score. Defaulting to neutral.")
		return 0.5
	except Exception as e:
		logging.error(f"Error analyzing sentiment for {symbol}: {e}")
		return 0.5





# Remove low-variance features

def remove_low_variance_features(features, threshold=0.01):
	"""Remove low-variance features."""
	selector = VarianceThreshold(threshold)
	selected_features = selector.fit_transform(features)
	selected_columns = features.columns[selector.get_support()]
	return pd.DataFrame(selected_features, columns=selected_columns)

# Remove highly correlated features
def remove_highly_correlated_features(df, threshold=0.95):
	"""Remove features with a high correlation to each other."""
	correlation_matrix = df.corr().abs()
	upper_triangle = correlation_matrix.where(
		np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
	)
	to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
	return df.drop(to_drop, axis=1)

def continual_model_optimization(data):
	"""Optimize model continually using advanced validation and fine-tuning."""
	try:
		# Split data
		features = data.drop(['timestamp', 'close'], axis=1)
		target = np.where(data['close'].shift(-1) > data['close'], 1, 0)

		# Train-test split
		tscv = TimeSeriesSplit(n_splits=5)
		best_model = None
		best_score = -float("inf")

		for train_idx, test_idx in tscv.split(features):
			X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
			y_train, y_test = target[train_idx], target[test_idx]

			# Perform hyperparameter tuning
			model = LGBMClassifier()
			param_grid = {
				'max_depth': [3, 5, 7],
				'learning_rate': [0.01, 0.1, 0.2],
				'n_estimators': [100, 200, 300],
			}
			grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=3)
			grid_search.fit(X_train, y_train)

			# Evaluate
			score = grid_search.best_score_
			if score > best_score:
				best_score = score
				best_model = grid_search.best_estimator_

		logging.info(f"Optimized Model Score: {best_score}")
		return best_model
	except Exception as e:
		logging.error(f"Model optimization error: {e}")
		return None
	
async def analyze_and_adjust_strategy(metrics, strategy):
	"""Adjust strategies based on deployment performance metrics."""
	try:
		logging.info(f"Analyzing metrics: {metrics}")
		if metrics['Sharpe Ratio'] < 1.0 or metrics['Max Drawdown'] > -0.2:
			logging.info("Strategy requires adjustment.")
			refined_strategy = await refined_strategy_with_gpt_retry(strategy)
			logging.info(f"Adjusted strategy: {refined_strategy}")
			return refined_strategy
		return strategy
	except Exception as e:
		logging.error(f"Error adjusting strategy: {e}")
		return strategy


def calculate_dynamic_risk(df, position):
	"""Calculate advanced risk management parameters."""
	try:
		atr = df['ATR'].iloc[-1]
		price = df['close'].iloc[-1]

		if position == "long":
			stop_loss = price - (2 * atr)
			take_profit = price + (4 * atr)
		elif position == "short":
			stop_loss = price + (2 * atr)
			take_profit = price - (4 * atr)
		else:
			stop_loss, take_profit = None, None

		return stop_loss, take_profit
	except Exception as e:
		logging.error(f"Error calculating risk parameters: {e}")
		return None, None


# Machine Learning Model Training
def train_model(data):
	"""
	Train the LightGBM model with preprocessing steps.

	Args:
		data (pd.DataFrame): Historical data for training.

	Returns:
		model: Trained LightGBM model.
	"""
	try:
		# Preprocess data
		features = data.drop(['timestamp', 'close'], axis=1)
		target = (data['close'].shift(-1) > data['close']).astype(int)

		features = normalize_data(features)
		features, target = balance_data(features, target)

		# Split dataset
		X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

		# Train model
		model = initialize_lgbm_model()
		model.fit(X_train, y_train)
		logging.info("Model trained successfully.")

		# Test predictions
		accuracy = model.score(X_test, y_test)
		logging.info(f"Model accuracy on sample dataset: {accuracy:.2f}")

		return model
	except Exception as e:
		logging.error(f"Error in train_model: {e}")
		return None



def preprocess_features(data):
	"""
	Preprocess features by removing low-variance and highly correlated features.

	Args:
		data (pd.DataFrame): Input data.

	Returns:
		pd.DataFrame: Preprocessed features.
	"""
	features = data.drop(['timestamp', 'close'], axis=1)

	# Remove low-variance features
	features = remove_low_variance_features(features)

	# Remove highly correlated features
	features = remove_highly_correlated_features(features)

	return features



async def test_gpt_strategy_refinement(strategy):
	"""
	Test GPT-based strategy refinement.

	Args:
		strategy (dict): Trading strategy.

	Returns:
		dict: Refined strategy.
	"""
	try:
		refined_strategy = await refined_strategy_with_gpt_retry(strategy)
		logging.info(f"GPT refined strategy: {refined_strategy}")
		return refined_strategy
	except Exception as e:
		logging.error(f"Error testing GPT strategy refinement: {e}")
		return strategy


async def log_gpt_refinements(strategy):
	"""
	Log GPT-based strategy refinements.

	Args:
		strategy (dict): Strategy to refine.

	Returns:
		dict: Refined strategy.
	"""
	refined_strategy = await test_gpt_strategy_refinement(strategy)
	with open("gpt_refinements.log", "a") as log_file:
		log_file.write(f"{datetime.now()}: {refined_strategy}\n")
	return refined_strategy



async def test_data_fetching(symbols):
	"""
	Test fetching data for Binance Futures symbols.

	Args:
		symbols (list): List of symbols to fetch.
	"""
	for symbol in symbols:
		data = await fetch_and_save_historical_data(symbol)
		logging.info(f"Fetched data for {symbol}: {len(data)} rows")


def check_time_alignment(dataframes):
	"""
	Ensure time alignment across multiple timeframes.

	Args:
		dataframes (dict): Mapping of timeframes to dataframes.

	Returns:
		bool: True if aligned, False otherwise.
	"""
	timestamps = [df['timestamp'] for df in dataframes.values()]
	return all(t.equals(timestamps[0]) for t in timestamps)


def update_prometheus_metrics(metrics):
	"""
	Update Prometheus metrics during backtesting and live trading.

	Args:
		metrics (dict): Performance metrics.
	"""
	pnl_metric.set(metrics["net_profit"])
	trade_count_metric.inc(metrics["total_trades"])
	logging.info(f"Updated Prometheus metrics: {metrics}")




def walk_forward_validation(model, data, target):
	tscv = TimeSeriesSplit(n_splits=5)
	scores = []
	for fold, (train_idx, test_idx) in enumerate(tscv.split(data)):
		X_train, X_test = data.iloc[train_idx], data.iloc[test_idx]
		y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
		model.fit(X_train, y_train)
		score = model.score(X_test, y_test)
		scores.append(score)
		logging.info(f"Fold {fold + 1}: Accuracy = {score:.2f}")
	average_score = sum(scores) / len(scores)
	logging.info(f"Average walk-forward validation accuracy: {average_score:.2f}")
	return average_score






def load_model_from_cloud():
	"""Load LightGBM model from Google Cloud."""
	try:
		model_file = "lightgbm_model.pkl"
		local_file = f"/tmp/{model_file}"

		# Download model from Google Cloud Storage
		bucket = storage_client.bucket(GCP_BUCKET_NAME)
		blob = bucket.blob(model_file)
		blob.download_to_filename(local_file)

		# Load the model
		with open(local_file, 'rb') as f:
			model = pickle.load(f)
		logging.info("Model loaded successfully from Google Cloud.")
		return model
	except Exception as e:
		logging.error(f"Error loading model: {e}")
		return None



class Strategy(bt.Strategy):
	params = (('ema_short', 20), ('ema_long', 50))

	def __init__(self):
		self.ema_short = bt.indicators.EMA(self.data.close, period=self.params.ema_short)
		self.ema_long = bt.indicators.EMA(self.data.close, period=self.params.ema_long)

	def next(self):
		if self.ema_short > self.ema_long and not self.position:
			self.buy(size=1)
		elif self.ema_short < self.ema_long and self.position:
			self.close()

def run_backtest(dataframe, strategy_fn, initial_balance=10000, fee=0.001):
	"""
	Backtest a trading strategy on historical data.
	Args:
		dataframe (pd.DataFrame): Historical market data.
		strategy_fn (callable): Function that generates trade signals (buy/sell/hold).
		initial_balance (float): Starting capital for the backtest.
		fee (float): Trading fee percentage.
	Returns:
		dict: Backtest results with performance metrics.
	"""
	balance = initial_balance
	position = 0
	entry_price = 0
	balance_history = []
	trades = []

	for i in range(len(dataframe)):
		signal = strategy_fn(dataframe.iloc[:i + 1])  # Generate signal up to the current point

		if signal == "buy" and position == 0:
			entry_price = dataframe['close'].iloc[i]
			position = balance / entry_price
			balance -= position * entry_price * (1 + fee)
			trades.append({"action": "buy", "price": entry_price, "amount": position})

		elif signal == "sell" and position > 0:
			exit_price = dataframe['close'].iloc[i]
			balance += position * exit_price * (1 - fee)
			trades.append({"action": "sell", "price": exit_price, "amount": position})
			position = 0

		balance_history.append(balance + (position * dataframe['close'].iloc[i] if position > 0 else 0))

	if position > 0:  # Close open position at the last price
		balance += position * dataframe['close'].iloc[-1]
		position = 0

	pnl_metric.set(balance - initial_balance)  # Update Prometheus PnL metric
	trade_count_metric.inc(len(trades))       # Update Prometheus trade count metric

	return {
		"final_balance": balance,
		"pnl": balance - initial_balance,
		"trades": trades
	}




def calculate_fibonacci_levels(df, period=50):
	"""Calculate Fibonacci retracement levels."""
	# Get the highest and lowest closing prices over the period
	highest_high = df['high'].rolling(window=period).max().iloc[-1]
	lowest_low = df['low'].rolling(window=period).min().iloc[-1]

	# Calculate the Fibonacci levels
	difference = highest_high - lowest_low
	level_23_6 = highest_high - (difference * 0.236)
	level_38_2 = highest_high - (difference * 0.382)
	level_50 = highest_high - (difference * 0.5)
	level_61_8 = highest_high - (difference * 0.618)

	# Return the Fibonacci levels
	return level_23_6, level_38_2, level_50, level_61_8



def feature_importance_analysis(model, X_train, y_train):
	result = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42, scoring='accuracy')
	feature_importances = result.importances_mean
	feature_names = X_train.columns

	# Plot feature importance
	plt.barh(feature_names, feature_importances)
	plt.xlabel("Feature Importance")
	plt.title("Feature Importance Analysis")
	plt.show()

	return feature_importances







def validate_trade_with_fibonacci(df, position):
	"""
	Validate trades using Fibonacci retracement levels for breakouts/reversals.
	This function expects the caller to pass a sliced DataFrame.
	"""
	try:
		# Ensure required columns exist
		required_columns = ['fib_23_6', 'fib_38_2', 'fib_50', 'fib_61_8', 'close']
		if not all(col in df.columns for col in required_columns):
			raise KeyError("Missing required columns in DataFrame for Fibonacci validation.")

		# Fetch latest Fibonacci levels and price
		close_price = df['close'].iloc[-1]
		fib_23_6 = df['fib_23_6'].iloc[-1]
		fib_38_2 = df['fib_38_2'].iloc[-1]
		fib_50 = df['fib_50'].iloc[-1]
		fib_61_8 = df['fib_61_8'].iloc[-1]

		# Validate position
		if position == "long":
			return fib_61_8 < close_price < fib_50  # Long: Price between support levels
		elif position == "short":
			return fib_23_6 > close_price > fib_38_2  # Short: Price between resistance levels

		return False
	except KeyError as e:
		logging.error(f"Error in Fibonacci validation: {e}")
		return False
	except Exception as e:
		logging.error(f"Unexpected error in Fibonacci validation: {e}")
		return False





#Calculate Indicators
def calculate_indicators(df):
	"""
	Optimized indicator calculations with vectorized operations.
	"""
	from ta.trend import EMAIndicator
	from ta.momentum import RSIIndicator
	from ta.volatility import AverageTrueRange

	df['EMA_20'] = EMAIndicator(df['close'], window=20).ema_indicator()
	df['EMA_50'] = EMAIndicator(df['close'], window=50).ema_indicator()
	df['RSI'] = RSIIndicator(df['close'], window=14).rsi()
	df['ATR'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
	df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

	# Calculate Fibonacci levels
	df['high_rolling'] = df['high'].rolling(window=50).max()
	df['low_rolling'] = df['low'].rolling(window=50).min()

	# Ensure no NaN values in rolling calculations
	df = df.dropna(subset=['high_rolling', 'low_rolling']).copy()

	difference = df['high_rolling'] - df['low_rolling']
	df['fib_23_6'] = df['high_rolling'] - (difference * 0.236)
	df['fib_38_2'] = df['high_rolling'] - (difference * 0.382)
	df['fib_50'] = df['high_rolling'] - (difference * 0.5)
	df['fib_61_8'] = df['high_rolling'] - (difference * 0.618)

	return df






#Fetch Symbol Data In All Time Frames
async def fetch_symbol_data_for_all_timeframes(symbol, timeframes, limit=500):
	"""
	Fetch historical data for a symbol across all specified timeframes.

	Args:
		symbol (str): The trading symbol (e.g., 'BTC/USDT').
		timeframes (list): List of timeframes to fetch (e.g., ['1m', '1h', '1d']).
		limit (int): Number of data points to fetch for each timeframe (default: 500).

	Returns:
		dict: A dictionary of timeframe to DataFrame mappings for the given symbol.
	"""
	async def fetch_data_for_timeframe(timeframe):
		try:
			ohlcv = await binance.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
			df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
			df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
			return timeframe, df
		except Exception as e:
			logging.error(f"Error fetching data for {symbol} on timeframe {timeframe}: {e}")
			return timeframe, pd.DataFrame()  # Return empty DataFrame on error

	tasks = [fetch_data_for_timeframe(tf) for tf in timeframes]
	results = await asyncio.gather(*tasks)
	return {tf: df for tf, df in results if not df.empty}


  
# Fetch Symbol Data
async def fetch_all_symbols_data_for_futures(symbols, timeframes, limit=500):
	"""
	Fetch historical data for all symbols across specified timeframes.

	Args:
		symbols (list): List of trading symbols (e.g., ['BTC/USDT', 'ETH/USDT']).
		timeframes (list): List of timeframes to fetch (e.g., ['1m', '1h', '1d']).
		limit (int): Number of data points to fetch for each timeframe (default: 500).

	Returns:
		dict: A dictionary of symbol to timeframe-to-DataFrame mappings.
	"""
	async def fetch_symbol_data(symbol):
		if not await validate_symbol(symbol):
			logging.warning(f"Skipping invalid symbol: {symbol}")
			return symbol, {}

		logging.info(f"Fetching data for {symbol} across timeframes: {timeframes}")
		data = await fetch_symbol_data_for_all_timeframes(symbol, timeframes, limit)
		return symbol, data

	tasks = [fetch_symbol_data(symbol) for symbol in symbols]
	results = await asyncio.gather(*tasks)
	return {symbol: data for symbol, data in results if data}


def evaluate_readiness_for_live_trading(backtest_results, model_accuracy, sharpe_threshold=1.5, max_drawdown_threshold=-0.1, min_accuracy=0.75):
	"""
	Evaluate whether the bot is ready for live trading based on backtesting results and ML performance.

	Args:
		backtest_results (dict): Backtesting results for each symbol and timeframe.
		model_accuracy (float): Accuracy of the ML model.
		sharpe_threshold (float): Minimum acceptable Sharpe Ratio.
		max_drawdown_threshold (float): Maximum acceptable drawdown.
		min_accuracy (float): Minimum acceptable ML model accuracy.

	Returns:
		bool: True if ready for live trading, False otherwise.
	"""
	for key, result in backtest_results.items():
		if result['sharpe_ratio'] < sharpe_threshold:
			logging.warning(f"Sharpe Ratio for {key} below threshold: {result['sharpe_ratio']} < {sharpe_threshold}")
			return False
		if result['max_drawdown'] < max_drawdown_threshold:
			logging.warning(f"Max Drawdown for {key} exceeds threshold: {result['max_drawdown']} < {max_drawdown_threshold}")
			return False

	if model_accuracy < min_accuracy:
		logging.warning(f"ML model accuracy below threshold: {model_accuracy} < {min_accuracy}")
		return False

	logging.info("Bot is ready for live trading.")
	return True





async def apply_best_strategy(experiment_results, live_strategy):
	"""
	Update the live strategy based on the best-performing backtested strategy.

	Args:
		experiment_results (pd.DataFrame): DataFrame of backtesting results.
		live_strategy (callable): Current live trading strategy function.

	Returns:
		callable: Updated live trading strategy.
	"""
	try:
		best_strategy = experiment_results.loc[experiment_results['sharpe_ratio'].idxmax()]
		if best_strategy['sharpe_ratio'] > 1.5:  # Define a threshold
			logging.info(f"Applying best strategy: {best_strategy['strategy']}")
			return best_strategy['strategy_function']
		return live_strategy
	except Exception as e:
		logging.error(f"Error applying best strategy: {e}")
		return live_strategy







async def adjust_risk_based_on_sentiment(sentiment_score, base_risk=0.01):
	"""
	Adjust risk parameters based on sentiment score.

	Args:
		sentiment_score (float): Sentiment score between 0 (negative) and 1 (positive).
		base_risk (float): Base risk percentage.

	Returns:
		float: Adjusted risk percentage.
	"""
	risk_adjusted = base_risk * (1 + sentiment_score - 0.5)  # Adjust risk linearly
	return max(0.005, min(0.02, risk_adjusted))  # Clamp risk between 0.5% and 2%



async def dynamic_strategy_refinement(metrics, strategy_fn, threshold=0.5):
	"""
	Dynamically refine strategy based on performance metrics.

	Args:
		metrics (dict): Live trading metrics.
		strategy_fn (callable): Current strategy function.
		threshold (float): Sharpe ratio threshold for refinement.

	Returns:
		callable: Updated strategy function.
	"""
	if metrics['sharpe_ratio'] < threshold:
		logging.info("Refining strategy using GPT.")
		return await refined_strategy_with_gpt_retry(strategy_fn)
	return strategy_fn

from prometheus_client import Gauge

api_call_latency_metric = Gauge('api_call_latency', 'Latency of API calls in seconds')
decision_time_metric = Gauge('decision_time', 'Time taken for decision-making in seconds')

async def monitor_api_latency(api_call_fn):
	"""
	Measure and log the latency of an API call.

	Args:
		api_call_fn (callable): API call function to measure.
	"""
	import time
	start_time = time.time()
	await api_call_fn()
	api_call_latency_metric.set(time.time() - start_time)

async def measure_decision_time(strategy_fn, data):
	"""
	Measure the time taken for a strategy function to make a decision.

	Args:
		strategy_fn (callable): Trading strategy function.
		data (pd.DataFrame): DataFrame of market data.
	"""
	import time
	start_time = time.time()
	strategy_fn(data)
	decision_time_metric.set(time.time() - start_time)

from prometheus_client import Gauge



async def monitor_latency(api_call_fn):
	"""
	Measure and log the latency of an API call.

	Args:
		api_call_fn (callable): API call function to measure.
	"""
	import time
	start_time = time.time()
	await api_call_fn()
	latency_metric.set(time.time() - start_time)



async def log_decision_time(strategy_fn, data):
	"""
	Measure the time taken for a strategy function to make a decision.

	Args:
		strategy_fn (callable): Trading strategy function.
		data (pd.DataFrame): DataFrame of market data.
	"""
	import time
	start_time = time.time()
	strategy_fn(data)
	decision_time_metric.set(time.time() - start_time)


#Live Trading Loop
async def live_trading_loop(symbols, strategy_fn, start_balance, max_loss_percent=5):
	"""
	Live trading loop to analyze symbols, validate trades, execute them, and enforce loss limits.

	Args:
		symbols (list): List of trading symbols.
		strategy_fn (callable): Trading strategy function.
		start_balance (float): Balance at the start of the day.
		max_loss_percent (float): Maximum allowable daily loss percentage.
	"""
	balance = start_balance
	logging.info("Starting live trading loop...")

	while True:
		try:
			# Enforce daily loss limit
			if await enforce_daily_loss_limit(start_balance, balance, max_loss_percent):
				logging.error("Daily loss limit exceeded. Stopping trading for the day.")
				break

			# Analyze symbols for trading opportunities
			logging.info("Analyzing trading opportunities...")
			top_trades = await analyze_symbols(symbols, strategy_fn, balance)

			for trade in top_trades:
				symbol = trade['symbol']
				df = trade['data']
				position = trade['position']

				# Validate trade and check if already in position
				if await has_minimum_position(symbol, position):
					logging.info(f"Skipping trade for {symbol}: Position already exists.")
					continue

				# Calculate trade amount and leverage
				trade_amount, leverage = await adjust_leverage_and_amount(symbol, balance)
				if trade_amount == 0:
					logging.warning(f"Trade amount is 0 for {symbol}. Skipping trade.")
					continue

				# Define stop loss and take profit dynamically
				stop_loss, take_profit = calculate_dynamic_risk_params(df, position)

				# Execute the trade
				trade_result = await execute_trade(symbol, position, trade_amount, stop_loss, take_profit, leverage, balance)
				logging.info(f"Executed trade for {symbol}: {trade_result}")

				# Update Prometheus metrics
				pnl_metric.set(trade_result.get('pnl', 0))  # Update PnL metric
				trade_count_metric.inc()  # Increment trade count metric

			# Update balance after all trades
			balance = (await binance.fetch_balance())['total']['USDT']
			logging.info(f"Updated USDT balance: {balance}")

			# Pause before the next iteration
			await asyncio.sleep(60)

		except Exception as e:
			logging.error(f"Error in live trading loop: {e}")
			break

	# End of trading loop
	logging.info("Exiting live trading loop. Closing resources...")
	await binance.close()
	scheduler.shutdown()


#Fetch Raw Data
import pandas as pd
import logging
from binance.client import Client

# Binance client initialization (replace with your API key and secret)
binance_api_key = "NsQEHYQ4oNqUT2p1n2iCl0rTUbssad2zOghTDh4byl8O57HTtl4NezES93nJl5Jf5cfT1ug"
binance_api_secret = "PSiaHxR0aswjlsHGfqovw46PyasdsN1E9px658yu6zZRCfKnPTwmoxf9X68nMOXYT04bf"
binance_client = Client(binance_api_key, binance_api_secret)

async def fetch_raw_data(symbol, interval='1h', limit=500):
	"""
	Fetch raw historical data from Binance API.

	Args:
		symbol (str): Trading pair symbol (e.g., "BTC/USDT").
		interval (str): Time interval for the data (default: '1h').
		limit (int): Number of data points to fetch (default: 500).

	Returns:
		pd.DataFrame: DataFrame containing raw historical data.
	"""
	try:
		# Convert symbol format (if necessary)
		binance_symbol = symbol.replace('/', '')

		logging.info(f"Fetching raw data for {symbol}...")

		# Fetch historical data from Binance
		klines = binance_client.get_klines(symbol=binance_symbol, interval=interval, limit=limit)

		# Convert to DataFrame
		columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 
				   'close_time', 'quote_asset_volume', 'number_of_trades', 
				   'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore']

		df = pd.DataFrame(klines, columns=columns)

		# Convert data types
		df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
		df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
		df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

		# Keep only necessary columns
		df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]

		logging.info(f"Successfully fetched raw data for {symbol}.")
		return df

	except Exception as e:
		logging.error(f"Error fetching raw data for {symbol}: {e}")
		return None



# Fetch Historical Data
import os
import pandas as pd

async def fetch_and_save_historical_data(symbol, timeframe):
	"""
	Fetch and prepare historical data for a given symbol and timeframe.

	Args:
		symbol (str): Trading pair symbol.
		timeframe (str): Timeframe for the data (e.g., '1h').

	Returns:
		pd.DataFrame: Prepared historical data or None if data is invalid.
	"""
	try:
		logging.info(f"Fetching historical data for {symbol} ({timeframe})...")
		df = await fetch_raw_data(symbol, timeframe)  # Fetch raw data
		if df is None or df.empty:
			logging.warning(f"No data found for {symbol} ({timeframe}). Skipping.")
			return None

		# Prepare the data
		df = prepare_data(df)

		# Validate required columns
		required_columns = ['EMA_20', 'EMA_50', 'fib_23_6', 'fib_38_2', 'fib_50', 'fib_61_8', 'close']
		missing_columns = [col for col in required_columns if col not in df.columns]
		if missing_columns:
			logging.warning(f"Missing columns in prepared data for {symbol} ({timeframe}): {missing_columns}")
			return None

		logging.info(f"Historical data for {symbol} ({timeframe}) prepared successfully.")
		return df
	except Exception as e:
		logging.error(f"Error fetching or preparing data for {symbol} ({timeframe}): {e}")
		return None












def fetch_historical_data_sync(symbol, timeframe='1h', limit=500):
	try:
		loop = asyncio.get_running_loop()
		return loop.create_task(fetch_and_save_historical_data(symbol, timeframe, limit))
	except RuntimeError as e:
		logging.error(f"Error in fetch_historical_data_sync: {e}")
		return pd.DataFrame()












# Adjust Leverage and Trade Amount
async def adjust_leverage_and_amount(symbol, balance):
	"""Adjust leverage and trade amount to meet Binance minimum requirements."""
	try:
		await binance.load_markets()
		market_info = binance.market(symbol)

		if not market_info:
			raise ValueError(f"Market info not found for {symbol}.")

		# Fetch minimums
		min_amount = market_info['limits']['amount']['min']
		min_notional = market_info['limits']['cost']['min']
		precision = market_info['precision']['amount']
		ticker = await binance.fetch_ticker(symbol)
		close_price = ticker.get('last')

		if not close_price:
			raise ValueError(f"Could not fetch price for {symbol}.")

		# Calculate trade amount
		trade_amount = max(min_amount, min_notional / close_price)
		trade_amount = round(trade_amount, precision)

		# Ensure trade amount meets the minimum
		if trade_amount * close_price < min_notional:
			logging.warning(f"Trade amount {trade_amount} below minimum notional. Adjusting.")
			trade_amount = min_amount

		if trade_amount == 0:
			logging.error(f"Trade amount for {symbol} is zero after adjustments. Skipping trade.")
			return 0, 1

		leverage = min(10, balance / (trade_amount * close_price))
		return trade_amount, leverage
	except Exception as e:
		logging.error(f"Error adjusting leverage and trade amount for {symbol}: {e}")
		return 0, 1


#Validate Symbol
async def validate_symbol(symbol):
	"""
	Validate if the given symbol is listed in Binance Futures markets.

	Args:
		symbol (str): The trading symbol to validate (e.g., 'BTC/USDT').

	Returns:
		bool: True if the symbol is valid and listed on Binance Futures, False otherwise.
	"""
	try:
		await binance.load_markets()
		if symbol in binance.futures_markets:
			return True
		else:
			logging.warning(f"Symbol {symbol} is not listed in Binance Futures markets.")
			return False
	except Exception as e:
		logging.error(f"Error validating symbol {symbol}: {e}")
		return False



def trade_is_profitable(symbol):
	"""Mock function to check if a trade is profitable."""
	# Replace with actual logic based on historical data or other criteria
	return True


def validate_trade_with_technical_analysis(df):
	"""Validate trades using a combination of technical indicators."""
	try:
		last_row = df.iloc[-1]
		ema_crossover = last_row['EMA_20'] > last_row['EMA_50']  # Bullish crossover
		rsi_overbought = last_row['RSI'] > 70
		rsi_oversold = last_row['RSI'] < 30
		bullish_engulfing = last_row.get('bullish_engulfing', False)  # From candlestick patterns

		# Confirm trade based on conditions
		if ema_crossover and not rsi_overbought and bullish_engulfing:
			return "long"
		elif not ema_crossover and rsi_oversold:
			return "short"
		return None
	except Exception as e:
		logging.error(f"Error validating trade with technical analysis: {e}")
		return None

#ML Trade Validation
def validate_trade(df, position, ml_confidence, sentiment_score):
	"""Validate trades using technical indicators, ML confidence, and sentiment analysis."""
	try:
		last_row = df.iloc[-1]
		ema_crossover = last_row['EMA_20'] > last_row['EMA_50']
		rsi_valid = (last_row['RSI'] < 70 if position == 'long' else last_row['RSI'] > 30)
		macd_signal = last_row['MACD'] > 0 if position == 'long' else last_row['MACD'] < 0
		bollinger_band_valid = last_row['close'] < last_row['Bollinger_High'] if position == 'long' else last_row['close'] > last_row['Bollinger_Low']

		if not ema_crossover:
			return False

		if not rsi_valid:
			return False

		if not macd_signal:
			return False

		if not bollinger_band_valid:
			return False

		confidence_threshold = 0.7
		sentiment_threshold = 0.6

		return ml_confidence >= confidence_threshold and sentiment_score >= sentiment_threshold

	except Exception as e:
		logging.error(f"Trade validation error: {e}")
		return False








#Calculate Candlestick Patterns

def calculate_candlestick_patterns(df):
	"""Identify candlestick patterns."""
	try:
		required_columns = ['open', 'close']
		for col in required_columns:
			if col not in df.columns:
				raise KeyError(f"Missing required column: {col}")
		
		df['bullish_engulfing'] = (df['close'] > df['open']) & (df['close'].shift(1) < df['open'].shift(1))
		df['bearish_engulfing'] = (df['close'] < df['open']) & (df['close'].shift(1) > df['open'].shift(1))
		return df
	except Exception as e:
		logging.error(f"Error calculating candlestick patterns: {e}")
		return df


#Dynamic Risk Management
def calculate_dynamic_risk_params(df, position, balance, leverage=1, risk_percentage=0.02):
	"""
	Dynamically calculate stop-loss and take-profit levels based on ATR, balance, and leverage.

	Args:
		df (pd.DataFrame): Market data containing 'close' and 'ATR' columns.
		position (str): 'long' or 'short'.
		balance (float): Current account balance.
		leverage (int): Leverage used in the trade (default: 1).
		risk_percentage (float): Maximum risk as a percentage of balance (default: 2%).

	Returns:
		tuple: (stop_loss, take_profit)
	"""
	try:
		atr = df['ATR'].iloc[-1]
		price = df['close'].iloc[-1]

		# Calculate risk amount in monetary terms
		max_risk_amount = balance * risk_percentage

		# Determine raw stop-loss and take-profit levels based on ATR
		if position == "long":
			stop_loss = price - (2 * atr / leverage)
			take_profit = price + (4 * atr / leverage)
		elif position == "short":
			stop_loss = price + (2 * atr / leverage)
			take_profit = price - (4 * atr / leverage)
		else:
			raise ValueError("Position must be 'long' or 'short'.")

		# Adjust stop-loss if it risks more than the maximum allowed
		potential_loss = abs(price - stop_loss)
		if potential_loss > max_risk_amount:
			if position == "long":
				stop_loss = price - max_risk_amount
			elif position == "short":
				stop_loss = price + max_risk_amount

		return stop_loss, take_profit

	except KeyError as e:
		logging.error(f"KeyError in calculate_dynamic_risk_params: {e}")
		return None, None
	except Exception as e:
		logging.error(f"Error in calculate_dynamic_risk_params: {e}")
		return None, None












# Save Backtest Results To Google Cloud
async def save_backtest_results_to_cloud(results, filename="backtest_results.json"):
	"""Save backtest results to Google Cloud."""
	try:
		await async_save_to_google_cloud(results, filename)
		logging.info(f"Backtest results saved to Google Cloud: {filename}")
	except Exception as e:
		logging.error(f"Error saving backtest results to Google Cloud: {e}")


#Validate Dataset
def validate_dataset(df, retries=3, delay=2):
	"""
	Validate the dataset to ensure it is complete and retry if empty.

	Args:
		df (pd.DataFrame): The dataset to validate.
		retries (int): Maximum number of retries for empty datasets.
		delay (int): Delay between retries in seconds.

	Returns:
		bool: True if the dataset is valid, False otherwise.
	"""
	attempt = 0
	while attempt < retries:
		if df is None:
			logging.warning("Dataset is None.")
			return False

		if not isinstance(df, pd.DataFrame):
			logging.warning(f"Dataset is not a DataFrame. It is of type {type(df)}.")
			return False

		if df.empty:
			logging.warning("Dataset is empty. Retrying...")
			time.sleep(delay)
			attempt += 1
			continue

		required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
		missing_columns = [col for col in required_columns if col not in df.columns]
		if missing_columns:
			logging.warning(f"Missing required columns in dataset: {missing_columns}")
			return False

		return True

	logging.error("Dataset validation failed after retries.")
	return False



#Run Backtests
async def run_backtests(symbols, strategy_fn, timeframes, initial_balance=10000):
	"""
	Run backtests for the given symbols and strategy.

	Args:
		symbols (list): List of trading symbols.
		strategy_fn (callable): Trading strategy function.
		timeframes (list): List of timeframes to backtest.
		initial_balance (float): Starting balance.

	Returns:
		dict: Backtesting results.
	"""
	results = {}

	for symbol in symbols:
		for timeframe in timeframes:
			try:
				logging.info(f"Fetching data for backtesting: {symbol} on {timeframe}...")
				df = await fetch_and_save_historical_data(symbol, timeframe)

				if not validate_dataset(df):
					logging.warning(f"Invalid dataset for {symbol} on {timeframe}. Skipping.")
					continue

				logging.info(f"Running backtest for {symbol} on {timeframe}...")
				result = await enhanced_backtest_strategy(df, strategy_fn, initial_balance)
				results[(symbol, timeframe)] = result
				logging.info(f"Backtest completed for {symbol} on {timeframe}. Final Balance: {result['final_balance']}")
			except Exception as e:
				logging.error(f"Error during backtesting for {symbol} on {timeframe}: {e}")

	return results






def compare_backtest_performance(regression_results, original_results):
	"""
	Compare the performance of regression-based and original strategies.

	Args:
		regression_results (pd.DataFrame): Backtest results for regression-based strategy.
		original_results (pd.DataFrame): Backtest results for original strategy.

	Returns:
		dict: A dictionary summarizing the key metrics comparison.
	"""
	try:
		if regression_results.empty or original_results.empty:
			logging.warning("One or both backtest result DataFrames are empty.")
			return {}

		metrics_to_compare = [
			"net_profit",
			"sharpe_ratio",
			"max_drawdown",
			"win_rate",
		]

		comparison_summary = {}
		for metric in metrics_to_compare:
			if metric in regression_results.columns and metric in original_results.columns:
				regression_metric = regression_results[metric].mean()
				original_metric = original_results[metric].mean()
				comparison_summary[metric] = {
					"regression_mean": regression_metric,
					"original_mean": original_metric,
					"difference": regression_metric - original_metric,
				}
			else:
				logging.warning(f"Metric '{metric}' is missing in one of the DataFrames.")

		logging.info("Comparison of Backtest Performance:")
		for metric, values in comparison_summary.items():
			logging.info(
				f"{metric.capitalize()}: "
				f"Regression Mean = {values['regression_mean']:.4f}, "
				f"Original Mean = {values['original_mean']:.4f}, "
				f"Difference = {values['difference']:.4f}"
			)

		try:
			import matplotlib.pyplot as plt
			metrics = list(comparison_summary.keys())
			regression_means = [comparison_summary[metric]["regression_mean"] for metric in metrics]
			original_means = [comparison_summary[metric]["original_mean"] for metric in metrics]

			plt.bar(metrics, regression_means, alpha=0.6, label="Regression Strategy")
			plt.bar(metrics, original_means, alpha=0.6, label="Original Strategy")
			plt.ylabel("Mean Metric Value")
			plt.title("Comparison of Backtest Performance")
			plt.legend()
			plt.show()
		except ImportError:
			logging.warning("matplotlib not installed. Skipping visualization.")

		return comparison_summary
	except Exception as e:
		logging.error(f"Error comparing backtest performance: {e}")
		return {}











#Live Trading Loop
async def live_trading_loop(symbols, strategy_fn, balance):
	"""
	Live trading loop to analyze symbols, validate trades, and execute them.

	Args:
		symbols (list): List of trading symbols.
		strategy_fn (callable): Trading strategy function.
		balance (float): Current account balance.
	"""
	while True:
		try:
			# Analyze symbols for trading opportunities
			logging.info("Analyzing trading opportunities...")
			top_trades = await analyze_symbols(symbols, strategy_fn, balance)

			for trade in top_trades:
				symbol = trade['symbol']
				df = trade['data']
				position = trade['position']

				# Validate trade
				if await has_minimum_position(symbol, position):
					logging.info(f"Skipping trade for {symbol}: Position already exists.")
					continue

				# Calculate trade amount and leverage
				trade_amount, leverage = await adjust_leverage_and_amount(symbol, balance)

				# Define stop loss and take profit
				stop_loss, take_profit = calculate_dynamic_risk_params(df, position)

				# Execute the trade
				trade_result = await execute_trade(symbol, position, trade_amount, stop_loss, take_profit, leverage, balance)
				logging.info(f"Executed trade: {trade_result}")

			# Update balance after trades
			balance = (await binance.fetch_balance())['total']['USDT']

			# Pause before the next iteration
			await asyncio.sleep(60)

		except Exception as e:
			logging.error(f"Error in live trading loop: {e}")
			break



async def refine_live_strategy(experiment_results, current_live_strategy):
	"""
	Refine the live strategy based on the best-performing strategy from experiments.

	Args:
		experiment_results (pd.DataFrame): Results of backtesting experiments.
		current_live_strategy (callable): The current live strategy function.

	Returns:
		callable: Updated live strategy function.
	"""
	return await apply_best_strategy(experiment_results, current_live_strategy)





#Refine Strategy
async def refine_strategy_if_needed(strategy, backtest_results):
	"""Refine the strategy based on backtest performance metrics."""
	for symbol, metrics in backtest_results.items():
		if metrics['sharpe_ratio'] < 1.0 or metrics['max_drawdown'] < -0.2:
			logging.info(f"Refining strategy for {symbol} due to poor performance metrics...")
			strategy = await refined_strategy_with_gpt_retry(strategy)
			logging.info(f"Refined strategy: {strategy}")
	return strategy

#Track Live Performance
def track_live_performance(trades, balance):
	"""
	Track live performance metrics during trading.

	Args:
		trades (list): List of executed trades.
		balance (float): Current trading balance.

	Returns:
		dict: Updated metrics for live trading.
	"""
	metrics = calculate_performance_metrics(trades, balance)
	pnl_metric.set(metrics["net_profit"])  # Update Prometheus PnL metric
	trade_count_metric.inc(metrics["total_trades"])  # Update Prometheus trade count metric

	logging.info(f"Live Performance Metrics: {metrics}")
	if metrics["win_rate"] < 50 or metrics["profit_factor"] < 1.5:
		logging.warning("Live performance is below target! Consider stopping or adjusting strategy.")
	return metrics



# Execute Trade
# Prometheus Metrics Updates in execute_trade
async def execute_trade(symbol, position, trade_amount, stop_loss, take_profit, leverage, balance):
	"""
	Execute a trade and update Prometheus metrics.

	Args:
		symbol (str): The trading symbol.
		position (str): The trade position ('long' or 'short').
		trade_amount (float): Amount to trade.
		stop_loss (float): Stop-loss price.
		take_profit (float): Take-profit price.
		leverage (float): Leverage for the trade.
		balance (float): Current account balance.
	"""
	try:
		await binance.set_leverage(leverage, symbol)
		side = 'buy' if position == 'long' else 'sell'
		order = await binance.create_order(symbol, 'market', side, trade_amount)

		# Calculate PnL dynamically
		current_price = (await binance.fetch_ticker(symbol))['last']
		entry_price = current_price if position == 'long' else stop_loss
		pnl = (current_price - entry_price) * trade_amount * leverage if position == 'long' else \
			(entry_price - current_price) * trade_amount * leverage

		# Update Prometheus Metrics
		pnl_metric.set(pnl)
		trade_count_metric.inc()

		logging.info(f"Executed {side} trade for {symbol} with PnL: {pnl}.")
		return order
	except Exception as e:
		logging.error(f"Error executing trade for {symbol}: {e}")
		return None

async def fetch_sentiment_data_for_symbol(symbol):
	"""
	Fetch sentiment data for a given symbol (mocked for now).
	Returns:
		float: Sentiment score between -1 and 1.
	"""
	try:
		return await mock_sentiment_api_call(symbol)
	except Exception as e:
		logging.error(f"Error fetching sentiment data for {symbol}: {e}")
		return 0



from prometheus_client import Gauge

trade_pnl_metric = Gauge("trade_pnl", "Profit and Loss for trades")
trade_count_metric = Gauge("trade_count", "Number of trades executed")
sharpe_ratio_metric = Gauge('sharpe_ratio', 'Sharpe Ratio of trades')
win_rate_metric = Gauge('win_rate', 'Win rate of trades')
sentiment_metric = Gauge('sentiment_score', 'Average sentiment score')

def update_trade_metrics(trade_result):
	trade_pnl_metric.set(trade_result.get("pnl", 0))
	trade_count_metric.inc()




async def execute_trade_with_sentiment(symbol, balance, sentiment_score):
	"""
	Execute a trade using sentiment-adjusted risk parameters.

	Args:
		symbol (str): Trading symbol.
		balance (float): Account balance.
		sentiment_score (float): Sentiment score between 0 and 1.
	"""
	base_risk = 0.01  # Base risk as 1% of balance
	adjusted_risk = await adjust_risk_based_on_sentiment(sentiment_score, base_risk)
	logging.info(f"Adjusted risk for {symbol}: {adjusted_risk:.4f}")
	# Use adjusted_risk for position sizing or stop-loss calculations


async def execute_trade_with_retry(symbol, position, amount, retries=3, delay=2):
	"""Execute a trade with retries and slippage validation."""
	side = 'buy' if position == 'long' else 'sell'
	for attempt in range(retries):
		try:
			# Validate slippage
			ticker = await binance.fetch_ticker(symbol)
			close_price = ticker['last']
			acceptable_slippage = close_price * 0.01
			market_price = await binance.fetch_order_book(symbol)['asks'][0][0]

			if abs(market_price - close_price) > acceptable_slippage:
				logging.warning(f"Slippage too high for {symbol}. Skipping trade.")
				return None

			# Execute trade
			order = await binance.create_order(symbol, 'market', side, amount)
			logging.info(f"Trade executed for {symbol}: {order}")
			return order
		except Exception as e:
			logging.warning(f"Trade execution failed for {symbol} (Attempt {attempt + 1}): {e}")
			await asyncio.sleep(2 ** attempt)  # Exponential backoff
	logging.error(f"Failed to execute trade for {symbol} after {retries} attempts.")
	return None






async def enforce_daily_loss_limit(start_balance, current_balance, max_loss_percent):
	loss = ((start_balance - current_balance) / start_balance) * 100
	if loss >= max_loss_percent:
		logging.error(f"Daily loss limit reached: {loss:.2f}%. Trading halted.")
		trade_pnl_metric.set(-loss)  # Update Prometheus metric
		return True
	logging.info(f"Current loss: {loss:.2f}%. Within limits.")
	return False




async def enforce_loss_protection():
	"""Check and close positions if loss exceeds 0.001% of entry price."""
	try:
		open_positions = await get_open_positions()
		for position in open_positions:
			symbol = position['symbol']
			if symbol not in position_data:
				continue

			# Get current price and entry price
			entry_price = position_data[symbol]['entry_price']
			current_price = (await binance.fetch_ticker(symbol))['last']

			# Calculate loss as a percentage
			loss_percentage = ((entry_price - current_price) / entry_price) * 100

			# Check if loss exceeds threshold
			if loss_percentage > 0.001:
				logging.info(f"Loss protection triggered for {symbol}. Closing position.")
				await binance.create_order(symbol, 'market', 'sell' if position_data[symbol]['position'] == 'long' else 'buy', position_data[symbol]['trade_amount'])
				del position_data[symbol]  # Remove closed position
	except Exception as e:
		logging.error(f"Error in loss protection: {e}")



logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s',
	handlers=[
		logging.FileHandler("trading_log.log"),
		logging.StreamHandler()
	]
)


# Analyze Symbol
async def analyze_symbols(symbols, model, balance):
	"""
	Analyze all symbols and prioritize the top trades across multiple timeframes.

	Args:
		symbols (list): List of trading symbols.
		model: Machine learning model for prediction.
		balance (float): Current trading balance.

	Returns:
		list: List of prioritized trade candidates.
	"""
	try:
		if model is None:
			logging.error("Model is not initialized. Skipping analysis.")
			return []

		logging.info("Fetching historical data for analysis...")
		symbol_data = await fetch_all_symbols_data_for_futures(symbols)

		trade_candidates = []

		for symbol, df in symbol_data.items():
			if df is None:
				logging.warning(f"No valid data for symbol {symbol}. Skipping analysis.")
				continue

			try:
				df = calculate_indicators(df)
				position = analyze_market_sentiment(symbol)
				if position == "hold":
					continue

				# Extract features for ML prediction
				features = df.drop(['timestamp', 'open', 'high', 'low', 'close', 'volume'], axis=1, errors='ignore')
				features = features.iloc[-1:]  # Use the latest data point
				probabilities = model.predict_proba(features)[-1]
				long_probability = probabilities[1]

				sentiment_score = await analyze_market_sentiment(symbol)
				is_valid = validate_trade(features, position, long_probability, sentiment_score)

				if is_valid:
					trade_candidates.append({
						"symbol": symbol,
						"probability": long_probability,
						"position": position
					})

			except Exception as e:
				logging.error(f"Error processing symbol {symbol}: {e}")

		return sorted(trade_candidates, key=lambda x: x["probability"], reverse=True)[:5]

	except Exception as e:
		logging.error(f"Error analyzing symbols: {e}")
		return []

















# Calculate Risk Parameters
def calculate_dynamic_risk_params(df, position):
	"""Calculate stop-loss and take-profit dynamically based on ATR and market conditions."""
	try:
		atr = df['ATR'].iloc[-1]
		close_price = df['close'].iloc[-1]

		# Risk/reward ratio
		stop_loss_multiplier = 2 if atr > 0.02 else 1.5
		take_profit_multiplier = 3

		if position == 'long':
			stop_loss = close_price - (stop_loss_multiplier * atr)
			take_profit = close_price + (take_profit_multiplier * atr)
		else:
			stop_loss = close_price + (stop_loss_multiplier * atr)
			take_profit = close_price - (take_profit_multiplier * atr)

		return stop_loss, take_profit
	except Exception as e:
		logging.error(f"Error calculating risk parameters: {e}")
		return None, None


def simulate_trade_execution(price, amount, fee=0.001, slippage=0.01):
	executed_price = price * (1 + slippage)
	fees = executed_price * amount * fee
	return executed_price * amount - fees



#Intelligent Retratining
# Intelligent Retraining
# Define the retrain_model function at a global scope
async def retrain_model(symbols):
	"""
	Retrain the model using combined historical data from all symbols.

	Args:
		symbols (list): List of trading symbols (e.g., ['BTC/USDT', 'ETH/USDT']).

	Returns:
		None
	"""
	try:
		logging.info("Fetching and combining historical data for retraining...")

		# Fetch and combine data for all symbols
		combined_data = pd.concat(
			[
				await fetch_and_save_historical_data(symbol)
				for symbol in symbols
				if validate_dataset(await fetch_and_save_historical_data(symbol))
			],
			ignore_index=True
		)

		# Check if the combined dataset is empty
		if combined_data.empty:
			logging.error("Combined dataset is empty. Skipping retraining.")
			return

		logging.info(f"Combined data size: {combined_data.shape}")

		# Check if data has changed significantly to warrant retraining
		if not await has_data_changed(combined_data, "historical_data_combined.csv"):
			logging.info("No significant data changes detected. Skipping retraining.")
			return

		logging.info("Training model on combined data...")

		# Train the model on the combined dataset
		model = train_model(combined_data)

		if model is None:
			logging.error("Model training failed. Skipping saving.")
			return

		# Save the model to cloud storage
		save_model_to_cloud(model, filename="retrained_model.pkl")

		logging.info("Model retrained and saved successfully.")

		# Optionally save the combined dataset for future reference
		await async_save_to_google_cloud(combined_data, "historical_data_combined.csv")
		logging.info("Combined dataset saved to cloud storage.")

	except Exception as e:
		logging.error(f"Error during model retraining: {e}")



#Schedule Retraining
def schedule_retraining(symbols):
	"""Schedule model retraining for all symbols."""
	scheduler.add_job(lambda: asyncio.create_task(retrain_model(symbols)), 'interval', days=1)



# Modify the schedule_retraining function
def schedule_retraining(symbols):
	"""Schedule model retraining for all symbols with a data freshness check."""
	scheduler.add_job(lambda: asyncio.create_task(retrain_model(symbols)), 'interval', days=1)



	
async def has_data_changed(new_data, cloud_filename="historical_data_combined.csv"):
	"""
	Check if the dataset has changed significantly compared to the saved version in the cloud.

	Args:
		new_data (pd.DataFrame): The new dataset to compare.
		cloud_filename (str): The filename of the existing dataset in cloud storage.

	Returns:
		bool: True if the data has changed significantly, False otherwise.
	"""
	try:
		# Load the existing dataset from the cloud
		existing_data = await async_load_from_google_cloud(cloud_filename)

		if existing_data is None or existing_data.empty:
			logging.info(f"No existing dataset found. Assuming data has changed.")
			return True

		# Calculate hash or similarity
		existing_hash = pd.util.hash_pandas_object(existing_data).sum()
		new_hash = pd.util.hash_pandas_object(new_data).sum()

		# Define a change threshold
		change_threshold = 0.05  # 5% change tolerance
		relative_change = abs(existing_hash - new_hash) / max(abs(existing_hash), 1)

		return relative_change > change_threshold
	except Exception as e:
		logging.error(f"Error checking data changes: {e}")
		return True






def sentiment_analysis_with_finbert(headlines):
	try:
		classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")
		sentiments = classifier(headlines)
		sentiment_scores = [1 if s['label'] == 'POSITIVE' else 0 for s in sentiments]
		return sum(sentiment_scores) / len(sentiment_scores)
	except Exception as e:
		logging.error(f"Error in sentiment analysis: {e}")
		return 0.5  # Neutral sentiment





def plot_performance(trades):
	plt.plot(trades['timestamp'], trades['pnl'], label="PnL")
	plt.xlabel('Time')
	plt.ylabel('Profit/Loss')
	plt.legend()
	plt.show()




def calculate_candlestick_patterns(df):
	df['bullish_engulfing'] = (df['close'] > df['open']) & (df['close'].shift(1) < df['open'].shift(1))
	return df

def calculate_vwap(df):
	"""Calculate the VWAP (Volume Weighted Average Price)."""
	try:
		if 'close' not in df or 'volume' not in df:
			raise KeyError("Missing 'close' or 'volume' column for VWAP calculation.")

		# Ensure no NaN values in 'close' or 'volume'
		if df[['close', 'volume']].isna().any().any():
			raise ValueError("NaN values detected in 'close' or 'volume'. Cannot calculate VWAP.")

		# Calculate VWAP
		df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
		return df
	except Exception as e:
		logging.error(f"Error calculating VWAP: {e}")
		return df




def add_time_features(df):
	try:
		# Ensure the 'timestamp' column is in datetime format
		df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
		
		# Drop rows where timestamp conversion failed (if any)
		df = df.dropna(subset=['timestamp']).copy()

		# Add time-based features
		df['day_of_week'] = df['timestamp'].dt.dayofweek
		df['hour'] = df['timestamp'].dt.hour
		return df
	except Exception as e:
		logging.error(f"Error in add_time_features: {e}")
		return df



async def process_symbols_concurrently(symbols, function, **kwargs):
	"""Run a function concurrently for multiple symbols."""
	tasks = [function(symbol, **kwargs) for symbol in symbols]
	return await asyncio.gather(*tasks)

# Example usage for fetching data
# results = await process_symbols_concurrently(usdt_symbols, fetch_and_save_historical_data, timeframe='1h')


async def get_open_positions():
	"""Fetch open positions from Binance Futures."""
	try:
		positions = await binance.fetch_positions()
		open_positions = [pos for pos in positions if float(pos['contracts']) > 0]
		return open_positions
	except Exception as e:
		logging.error(f"Error fetching open positions: {e}")
		return []


async def has_minimum_position(symbol, side):
	"""
	Check if a coin already has an open position with the minimum required amount for the given side (long/short).
	"""
	try:
		open_positions = await get_open_positions()
		for position in open_positions:
			if position['symbol'] == symbol:
				# Check if the position matches the requested side
				is_long = float(position['contracts']) > 0
				if (side == 'long' and is_long) or (side == 'short' and not is_long):
					trade_amount = abs(float(position['contracts']))  # Amount of the open position
					market_info = binance.market(symbol)
					min_amount = market_info['limits']['amount']['min']
					if trade_amount >= min_amount:
						logging.info(f"Position for {symbol} with side {side} already exists.")
						return True
		return False
	except Exception as e:
		logging.error(f"Error checking minimum position for {symbol} and side {side}: {e}")
		return False


# Main Function
# Replace the `while True` in the main function with a loop that listens for `shutdown_event`
# Main Function
# Imports
import asyncio
import logging
from prometheus_client import Gauge, start_http_server
import pandas as pd
import numpy as np
from binance.client import AsyncClient
from retrying import retry

# Constants
MAX_CONCURRENT_TASKS = 10
PROMETHEUS_PORT = 8000



# Function Definitions

# Enhancements applied: Retry mechanisms, better concurrency handling, logging improvements,
# Prometheus metrics, dynamic symbol validation, adaptive risk management, and more.

# [All code with the mentioned improvements]

# Main Function with All Fixes and Enhancements
from prometheus_client import Gauge, start_http_server




def aggregate_backtest_metrics(backtest_results) -> dict:
	"""
	Aggregates backtest metrics from the results.

	Args:
		backtest_results (pd.DataFrame): Backtest results for all strategies.

	Returns:
		dict: Aggregated metrics such as average PnL, win rate, and max drawdown.
	"""
	try:
		avg_pnl = backtest_results['pnl'].mean()
		win_rate = (backtest_results['wins'] / backtest_results['trades']).mean()
		max_drawdown = backtest_results['drawdown'].min()
		return {
			"average_pnl": avg_pnl,
			"win_rate": win_rate,
			"max_drawdown": max_drawdown,
		}
	except Exception as e:
		logging.error(f"Error aggregating backtest metrics: {e}")
		return {}


def collect_backtest_metrics() -> dict:
	"""
	Collects aggregated backtest performance metrics.

	Returns:
		dict: Dictionary of backtest performance metrics.
	"""
	global backtest_metrics
	return backtest_metrics


# Global variable to store backtest metrics
backtest_metrics = {}
# Main Function
async def main():
	"""
	Main function to execute the trading bot workflow with enhanced features.
	Combines regression trends, Prometheus metrics, sentiment analysis, GPT-based strategy refinement,
	concurrency optimization, dynamic strategy adjustments, and live trading readiness.
	"""
	global ml_model
	global backtest_metrics

	# Initialize Prometheus metrics
	initialize_prometheus(port=8000)
	live_balance = await get_live_balance("USDT")
	# Define parameters
	timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
	initial_balance = 10000
	# Adjust backtesting balance to match live balance if needed
	if live_balance == 0:
			logging.error("Live balance is zero. Exiting...")
			return  # Exit if no balance is available
		
	if initial_balance > live_balance:
		logging.warning(f"Backtesting initial balance ({initial_balance}) exceeds live balance ({live_balance}). Adjusting...")
		initial_balance = live_balance
	logging.info(f"Initial USDT balance: {live_balance}")
	logging.info(f"Adjusted backtesting balance: {initial_balance}")
	max_loss_percent = 5
	max_concurrent_tasks = 10  # Limit for concurrent API calls

	# Initialize Semaphore for rate-limiting API calls
	semaphore = asyncio.Semaphore(max_concurrent_tasks)

	logging.info("Starting the trading bot...")

	try:
		# Load market data and validate symbols
		logging.info("Fetching market data...")
		markets = await binance.load_markets()
		symbols = [
			symbol for symbol in markets
			if symbol.endswith('/USDT') and 'future' in markets[symbol]['type']
		]
		symbols = [symbol for symbol in symbols if await validate_symbol(symbol)]
		logging.info(f"Validated {len(symbols)} USDT futures symbols.")

		# Fetch current balance
		balance = (await binance.fetch_balance())['total']['USDT']
		logging.info(f"Initial USDT balance: {balance}")
		if initial_balance > balance:
			logging.warning(f"Backtesting initial balance ({initial_balance}) exceeds live balance ({balance}). Adjusting...")
			initial_balance = balance

		# Run backtests with regression integration
		logging.info("Running backtests with regression trends...")
		backtest_results = await run_backtest_experiments(symbols, timeframes, initial_balance)
		backtest_metrics = aggregate_backtest_metrics(backtest_results)
		logging.info("Backtests completed. Results saved.")

		# Evaluate performance with regression
		regression_results = backtest_results.loc[backtest_results['strategy'] == "Regression"]
		original_results = backtest_results.loc[backtest_results['strategy'] == "Original"]
		compare_backtest_performance(regression_results, original_results)

		# Refine strategies based on backtest results
		refined_strategy = await integrate_backtest_results(backtest_results)

		# Validate readiness for live trading
		model_accuracy = walk_forward_validation(ml_model, backtest_results)
		if not evaluate_readiness_for_live_trading(
			backtest_results,
			model_accuracy,
			sharpe_threshold=1.5,
			max_drawdown_threshold=-0.1,
			min_accuracy=0.75
		):
			logging.error("Trading bot is not ready for live trading. Exiting.")
			return

		# Save model and strategies
		save_model_to_cloud(ml_model)
		await async_save_to_google_cloud(refined_strategy, "refined_strategy.json")
		logging.info("Model and refined strategy saved to cloud.")

		# Start live trading
		logging.info("Starting live trading loop...")
		while True:
			try:
				# Perform trading logic
				trade_results = await live_trading_with_refinements(
					symbols, 
					balance, 
					max_loss_percent=max_loss_percent
				)

				# Dynamically trigger GPT-based strategy refinement
				if performance_drops_below_threshold():
					logging.info("Performance dropped below threshold. Triggering GPT-based strategy refinement.")
					refined_strategy = await refine_strategy_with_gpt()
					if refined_strategy:
						logging.info(f"Refined Strategy: {refined_strategy}")
						# Apply refined strategy adjustments dynamically
						await async_save_to_google_cloud(refined_strategy, "refined_strategy.json")

				# Pause for the next iteration
				await asyncio.sleep(60)  # Example pause for 1 minute

			except Exception as e:
				logging.error(f"Error in live trading loop: {e}")
				break

	except Exception as e:
		logging.error(f"Unexpected error in main: {e}")
		return

	finally:
		if scheduler.running:
			await scheduler.shutdown(wait=False)
			logging.info("Scheduler shut down.")






# Additional functions and recommendations integrated...











if __name__ == "__main__":
	try:
		asyncio.run(main())
	except (KeyboardInterrupt, SystemExit):
		logging.info("Application terminated.")



