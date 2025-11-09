import time
import pandas as pd
import numpy as np
import requests
from binance.client import Client
from typing import List, Dict, Callable, Tuple, Any


def get_klines_multi(client: Client, symbols: List[str], interval: str, limit: int = 500) -> Dict[str, pd.DataFrame]:
    """
    Fetch candlestick (kline) data from Binance for multiple symbols.

    Parameters:
        symbols (list of str): List of trading pair symbols, e.g. ['BTCUSDT', 'ETHUSDT'].
        interval (str): Kline interval, e.g. Client.KLINE_INTERVAL_15MINUTE, Client.KLINE_INTERVAL_1HOUR.
        limit (int): Optional. Number of candles to retrieve per symbol (default 500, max 1000).

    Returns:
        dict :Dictionary mapping each symbol to its corresponding pandas DataFrame.
    """
    data = {}

    start_time = time.time()

    for symbol in symbols:
        print(f"Fetching {symbol} ({interval}, limit={limit}) ...")
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        
        cols = ["open_time","open","high","low","close","volume","close_time",
                "quote_asset_volume","num_trades","taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume","ignore"]
        
        df = pd.DataFrame(klines, columns=cols)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        numeric_cols = ["open","high","low","close","volume"]
        df[numeric_cols] = df[numeric_cols].astype(float)
        df = df[["open_time","open","high","low","close","volume","close_time","num_trades"]]
        
        data[symbol] = df

    end_time = time.time()

    elapsed_time = end_time-start_time
    print(f"Elapsed Time: {elapsed_time:.2f}")
    
    return data


def apply_to_dict(stock_dict: Dict[str, pd.DataFrame], function: Callable[..., pd.DataFrame], **kwargs) -> Dict[str, pd.DataFrame]:
    """
    Apply a transformation function to each DataFrame in a dictionary.

    Args:
        stock_dict (Dict[str, pd.DataFrame]): Mapping of symbol → DataFrame.
        function (Callable[..., pd.DataFrame]): Function to apply to each DataFrame.
        **kwargs: Additional keyword arguments for the function.

    Returns:
        Dict[str, pd.DataFrame]: Updated mapping with transformed DataFrames.
    """

    new_stock_dict: Dict[str, pd.DataFrame] = {}
    for symbol, df in stock_dict.items():
        df_copy = df.copy()
        try:
            new_stock_dict[symbol] = function(df_copy, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Error processing {symbol}: {e}")
    return new_stock_dict


def calculate_ma(df: pd.DataFrame, period: int, source: str, method: str) -> pd.DataFrame:
    """
    Calculate a moving average (EMA or SMA) and add it as a new column.

    Args:
        df (pd.DataFrame): DataFrame containing a source column (e.g., 'Close').
        period (int): Lookback period for the moving average.
        source (str, optional): Column to calculate MA on. Defaults to "Close".
        method (str, optional): Type of moving average ("EMA" or "SMA"). Defaults to "EMA".

    Returns:
        pd.DataFrame: DataFrame with the new MA column added.

    Raises:
        ValueError: If the source column is missing, if period <= 0, or if method is invalid.
        TypeError: If period is not an integer.
    """
    try:
        if method.upper() == "EMA":
            ma_series = df[source].ewm(span=period, adjust=False).mean()
            ma_series.iloc[:period - 1] = pd.NA
        else:  # SMA
            ma_series = df[source].rolling(window=period).mean()

        col_name = f"{method.upper()}_{source}_{period}"
        df[col_name] = ma_series
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to calculate {method.upper()} on column '{source}' with period {period}: {e}")


def mark_crossovers(df: pd.DataFrame, short_ma_params: Tuple, long_ma_params: Tuple) -> pd.DataFrame:
    """
    Detect moving average crossovers (Bullish or Bearish).

    Args:
        df (pd.DataFrame): DataFrame containing the required MA columns.
        short_ma_params (Tuple[int, str, str]): (period, source, method) for the short MA.
        long_ma_params (Tuple[int, str, str]): (period, source, method) for the long MA.

    Returns:
        pd.DataFrame: DataFrame with an added crossover signal column.

    Raises:
        ValueError: If the required MA columns are missing or periods are invalid.
    """
    try:
        short_p, short_s, short_m = short_ma_params
        long_p, long_s, long_m = long_ma_params

        if short_p >= long_p:
            raise ValueError(
                f"Short MA must have smaller period than Long MA.\nFound Short MA Period: {short_p}, Long MA Period: {long_p}"
            )

        short_col = f"{short_m}_{short_s}_{short_p}"
        long_col = f"{long_m}_{long_s}_{long_p}"
        crossover_col = f"{short_col}_{long_col}_Crossover"

        if short_col not in df.columns or long_col not in df.columns:
            raise ValueError(
                f"Required columns '{short_col}' and/or '{long_col}' not found in DataFrame."
            )

        cross_up = (df[short_col] > df[long_col]) & (df[short_col].shift(1) <= df[long_col].shift(1))
        cross_down = (df[short_col] < df[long_col]) & (df[short_col].shift(1) >= df[long_col].shift(1))

        df[crossover_col] = "No"
        df.loc[cross_up, crossover_col] = "Bullish"
        df.loc[cross_down, crossover_col] = "Bearish"

        return df
    except Exception as e:
        raise RuntimeError(f"Failed to calculate MA Crossover: {e}")  


def add_indicators(df: pd.DataFrame, indicators_config: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Clean OHLCV data and compute technical indicators based on a config dictionary.

    Args:
        df (pd.DataFrame): Symbol DataFrame with OHLCV columns.
        indicators_config (Dict[str, Any], optional): Dictionary with indicator names as keys and parameters as values.

    Returns:
        pd.DataFrame: DataFrame with added indicators.
    """
    if indicators_config is None:
        raise ValueError("indicators_config must be provided as a dictionary.")

    # Ensure numeric OHLCV
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Moving averages
    for period, source, method in indicators_config.get("moving_averages", []):
        df = calculate_ma(df, period=period, source=source, method=method)
        ma_col = f"{method}_{source}_{period}"
        df[f"Above_{ma_col}"] = df["close"] > df[ma_col]

    # Crossover signals
    for short_params, long_params in indicators_config.get("crossovers", []):
        df = mark_crossovers(df, short_params, long_params)
   
    return df


def get_required_price_for_crossover(df: pd.DataFrame, ema_fast_col: str ="EMA_close_8", ema_slow_col: str ="EMA_open_20", period_fast: int =8, offset: float =1e-6):
    """
    Calculate price that is needed for crossover in the current candle.

    Args:
        df (pd.DataFrame): Symbol DataFrame with OHLCV columns.
        ema_fast_col (str): Columnt to take fast EMA from.
        ema_slow_col (str): Columnt to take slow EMA from.
        period_fast (int): Length of the fast period for required price calculation.
        offset (float): Small bias to estimate for cross up or below, not just equalizing of moving averages.

    Returns:
        float: Required price change for crossover.
    """
    ema_fast_latest = df[ema_fast_col].iloc[-2]
    ema_slow_latest = df[ema_slow_col].iloc[-1]
    ema_slow_prev = df[ema_slow_col].iloc[-2]

    threshold = ema_slow_latest
    alpha = 2 / (period_fast + 1)

    current_price = df["open"].iloc[-1]

    if ema_fast_latest < ema_slow_prev:
        desired_ema = threshold + offset
    else:
        desired_ema = threshold - offset

    needed_price = (desired_ema - (1 - alpha) * ema_fast_latest) / alpha
    required_percent_change = (needed_price - current_price) / current_price * 100

    return required_percent_change


def send_telegram_alert(msg: str, telegram_bot_token: str, chat_id: int):
    """
    Sends message via Telegram bot.

    Args:
        msg (str): Message to send.
        telegram_bot_token (str): Telegram bot token.
        chat_id (int): Chat ID.
    """

    url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": msg,
        "parse_mode": "HTML"
    }
    requests.post(url, json=payload)


def get_thresholds(interval: str, thresholds_dict: dict) -> float:
    """
    Return threshold for pairs in current interval.

    Args:
        interval (str): Interval to get thresholds for.
        thresholds_dict (dict): Dict with threhold

    Returns:
        dict: Dict with elements {pair: threshold}
    
    """
    interval_thresholds = thresholds_dict.get(interval, {})
    return interval_thresholds
