import os
import json
import logging
import boto3
import pandas as pd
from binance.client import Client
from functions import *


# Logging Setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment Variables
BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY")
BINANCE_API_SECRET = os.environ.get("BINANCE_SECRET_KEY")
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["CHAT_ID"]
THRESHOLD_BUCKET = os.environ["THRESHOLD_BUCKET"]
THRESHOLD_KEY = os.environ["THRESHOLD_KEY"]

if not BINANCE_API_KEY or not BINANCE_API_SECRET:
    raise RuntimeError(
        "ERROR: API keys not found. Set BINANCE_API_KEY and BINANCE_SECRET_KEY."
    )

# Binance Client
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

# S3
output_bucket = "crypto-potential-ema-alerts"
s3 = boto3.client("s3")

# Parameters
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]

INTERVAL_MAP = {
    "15m": Client.KLINE_INTERVAL_15MINUTE,
    "1h": Client.KLINE_INTERVAL_1HOUR,
    "4h": Client.KLINE_INTERVAL_4HOUR,
}

LIMIT = 21

# Indicator configuration
DEFAULT_INDICATORS = {
    "moving_averages": [
        [8, "close", "EMA"],
        [20, "open", "EMA"],
    ],
    "crossovers": [
        [[8, "close", "EMA"], [20, "open", "EMA"]]
    ],
}


def lambda_handler(event, context):

    logger.info(f"Incoming event: {json.dumps(event)}")

    # Parse interval input
    interval_key = event.get("interval", "15m")
    if interval_key not in INTERVAL_MAP:
        logger.warning(f"Invalid interval '{interval_key}' — defaulting to 15m")
        interval_key = "15m"

    interval = INTERVAL_MAP[interval_key]
    logger.info(f"Selected interval: {interval_key}")

    # Fetch data from Binance
    try:
        data = get_klines_multi(
            client=client,
            symbols=SYMBOLS,
            interval=interval,
            limit=LIMIT
        )
        logger.info(f"Fetched data for symbols: {list(data.keys())}")
        if not data:
            logger.error("No data returned from Binance.")
            return {"status": "error", "reason": "no_data_from_binance"}
    except Exception as e:
        logger.exception("Error fetching data from Binance")
        return {"status": "error", "reason": str(e)}

    # Process indicators
    try:
        data_labeled = apply_to_dict(
            data, 
            add_indicators, 
            indicators_config=DEFAULT_INDICATORS
        )
        logger.info("Indicators applied successfully.")
    except Exception as e:
        logger.exception("Error applying indicators")
        return {"status": "error", "reason": "indicator_processing_failed"}

    # Calculate required price for crossover
    try:
        required_price_json = apply_to_dict(data_labeled, get_required_price_for_crossover)
    except Exception as e:
        logger.exception("Error calculating required crossover prices")
        return {"status": "error", "reason": "crossover_calc_failed"}
    
    logger.info(f"Raw crossover JSON: {required_price_json}")

    # Convert to DataFrame
    required_price_df = (
        pd.DataFrame.from_dict(
            required_price_json, orient="index", columns=["required_change_pct"]
        )
        .reset_index()
        .rename(columns={"index": "pair"})
    )

    # Map thresholds
    try:
        obj = s3.get_object(Bucket=THRESHOLD_BUCKET, Key=THRESHOLD_KEY)
        body = obj["Body"].read().decode("utf-8")
        alert_thresholds = json.loads(body)
        alert_thresholds = alert_thresholds.get(interval_key)
    except Exception as e:
        logger.exception("Failed to load thresholds from S3")
        return {"status": "error", "reason": "thresholds_load_failed"}

    required_price_df["alert_threshold"] = required_price_df["pair"].map(alert_thresholds)

    # Filter
    alert_df = required_price_df[
        required_price_df["alert_threshold"].notna()
        & (required_price_df["required_change_pct"].abs() <= required_price_df["alert_threshold"])
    ]

    # Prepare Telegram message
    if alert_df.empty:
        logger.info("No alerts to send.")
        return {"ok": True, "alerts": 0}

    message_content = "\n".join(
        f"{row['pair']} is within {row['required_change_pct']:.2f}% of crossover"
        for _, row in alert_df.iterrows()
    )

    message_content = f"{interval_key} interval\n\n" + message_content
    
    # Send via Telegram
    try:
        send_telegram_alert(message_content, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        logger.info("Telegram message sent.")
    except Exception:
        logger.exception("Telegram send failed")
        return {"status": "error", "reason": "telegram_failed"}

    return {"ok": True, "alerts": len(alert_df)}
