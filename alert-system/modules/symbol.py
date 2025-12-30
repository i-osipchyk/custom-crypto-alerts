import time
import json
import aiofiles
import pandas as pd
from datetime import datetime, timedelta, timezone
from modules.config import UP_PCT_THRESHOLD_DAY, UP_PCT_THRESHOLD_NOW, DAILY_VOLUME_THRESHOLD, ALERT_COOLDOWN_MS, JSON_ALERTS_FILE, JSON_ALERTS_FILE_LOCK


class Symbol:
    def __init__(self, symbol, logger):
        self.symbol = symbol
        self.logger = logger

        self.open_price = None
        self.threshold_price_day = None
        self.threshold_price_now = None
        self.was_up_today = False
        self.is_up = False
        
        self.df_5m = pd.DataFrame()

        self.daily_volume_usdt = 0

        self.alerts = {
            "vwap": None,
            "ema20": None
        }

        self.session_date = datetime.now(timezone.utc).date()
        self.session_start_ms = int(
            datetime.now(timezone.utc)
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .timestamp() * 1000
        )

    async def init_df(self, session):
        # start of today (UTC)
        now = datetime.now(timezone.utc)
        today_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        start_ms = int(today_start.timestamp() * 1000)

        url = (
            "https://fapi.binance.com/fapi/v1/klines"
            f"?symbol={self.symbol.upper()}"
            f"&interval=5m"
            f"&startTime={start_ms}"
            f"&limit=1000"
        )

        async with session.get(url) as resp:
            data = await resp.json()

            if not isinstance(data, list) or len(data) == 0:
                self.logger.error(
                    f"Failed to fetch klines for {self.symbol}. Response: {data}"
                )
                return

            try:
                rows = []
                for k in data:
                    rows.append({
                        "t": int(k[0]),
                        "open_time": datetime.fromtimestamp(int(k[0]) / 1000, tz=timezone.utc),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                        "volume_usdt": float(k[7]),
                    })

                self.df_5m = pd.DataFrame(rows)

                # ensure sorted & typed
                self.df_5m.sort_values("t", inplace=True)
                self.df_5m.reset_index(drop=True, inplace=True)
                self.open_price = self.df_5m.at[0, "open"]
                self.threshold_price_day = self.open_price * (1 + UP_PCT_THRESHOLD_DAY / 100)
                self.threshold_price_now = self.open_price * (1 + UP_PCT_THRESHOLD_NOW / 100)
                
                self.daily_volume_usdt = self.df_5m["volume_usdt"].sum()

                self.alerts["vwap"] = int(k[0])
                self.alerts["ema20"] = int(k[0])

                self.logger.info(
                    f"{self.symbol}: Initialized df_5m with {len(self.df_5m)} candles. Current volume: {self.daily_volume_usdt}"
                )

            except (IndexError, KeyError, ValueError, TypeError) as e:
                self.logger.error(
                    f"Error parsing klines for {self.symbol}: {e} | Response: {data}"
                )

    async def update_from_kline(self, k):
        t = int(k["t"])
        open_time = datetime.fromtimestamp(t / 1000, tz=timezone.utc)
        open_ = round(float(k["o"]), 6)
        high = round(float(k["h"]), 6)
        low = round(float(k["l"]), 6)
        close = round(float(k["c"]), 6)
        volume = round(float(k["v"]), 6)
        volume_usdt = round(float(k["q"]), 6)
        
        current_date = open_time.date()

        if current_date != self.session_date:
            self.logger.info(f"{self.symbol}: New UTC day detected")
            self.reset_daily_state(open_time, open_)

        # check pct up
        if not self.was_up_today and high > self.threshold_price_day:
            self.was_up_today = True
          
        self.is_up = close > self.threshold_price_now


        # empty dataframe
        if self.df_5m.empty:
            self.df_5m.loc[0] = {
                "t": t,
                "open_time": open_time,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                'volume_usdt': volume_usdt,
                "EMA_9": None,
                "EMA_20": None,
                "VWAP": None,
            }
            return
        
        # new day reset
        if self.open_price is None:
            self.open_price = open_
            self.threshold_price_day = self.open_price * (1 + UP_PCT_THRESHOLD_DAY / 100)
            self.threshold_price_now = self.open_price * (1 + UP_PCT_THRESHOLD_NOW / 100)

            self.logger.info(
                f"{self.symbol}: New day open={self.open_price}, "
                f"day_threshold={self.threshold_price_day:.6f}, "
                f"now_threshold={self.threshold_price_now:.6f}"
            )

        last_idx = self.df_5m.index[-1]
        last_t = self.df_5m.at[last_idx, "t"]

        # same candle (partial update)
        if t == last_t:
            self.df_5m.loc[last_idx, ["open", "high", "low", "close", "volume", "volume_usdt"]] = [
                open_, high, low, close, volume, volume_usdt
            ]

        # new candle
        elif t > last_t:
            self.df_5m.loc[len(self.df_5m)] = {
                "t": t,
                "open_time": open_time,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "volume_usdt": volume_usdt,
                "EMA_9": None,
                "EMA_20": None,
                "VWAP": None,
            }

        # EMAs
        closes = self.df_5m["close"]

        self.ema_9 = closes.ewm(span=9, adjust=False).mean().iloc[-1]
        self.ema_20 = closes.ewm(span=20, adjust=False).mean().iloc[-1]

        # VWAP (session only)
        session_df = self.df_5m[self.df_5m["t"] >= self.session_start_ms]

        tp = (session_df["high"] + session_df["low"] + session_df["close"]) / 3
        self.vwap = (tp * session_df["volume_usdt"]).sum() / session_df["volume_usdt"].sum()

        # daily volume (session only)
        self.daily_volume_usdt = session_df["volume_usdt"].sum()

        # send alert
        if self.was_up_today and self.is_up and self.daily_volume_usdt > DAILY_VOLUME_THRESHOLD:
            if close > self.vwap and close < self.vwap * 1.005:
                await self.send_alert(close, "vwap")
            if close > self.ema_20 and close < self.ema_20 * 1.005:
                await self.send_alert(close, "ema20")

    def reset_daily_state(self, open_time: datetime, open_: float):
        self.logger.info(f"{self.symbol}: Resetting daily state")

        self.session_date = open_time.date()
        self.session_start_ms = int(
            open_time.replace(hour=0, minute=0, second=0, microsecond=0)
            .timestamp() * 1000
        )

        self.df_5m = self.df_5m.tail(21)

        self.open_price = open_
        self.threshold_price_day = open_ * (1 + UP_PCT_THRESHOLD_DAY / 100)
        self.threshold_price_now = open_ * (1 + UP_PCT_THRESHOLD_NOW / 100)

        self.was_up_today = False
        self.is_up = False

        self.daily_volume_usdt = 0

        self.vwap = None

        # self.df_5m.at[self.df_5m.index[-1], "EMA_9"] = round(self.ema_9, 6)
        # self.df_5m.at[self.df_5m.index[-1], "EMA_20"] = round(self.ema_20, 6)
        # self.df_5m.at[self.df_5m.index[-1], "VWAP"] = round(self.vwap, 6)

    async def send_alert(self, price, reason):
        now_ms = int(time.time() * 1000)

        last_alert = self.alerts.get(reason)
        if last_alert and (now_ms - last_alert) < ALERT_COOLDOWN_MS:
            return
        
        alert = {
            "symbol": self.symbol,
            "time": datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc).isoformat(),
            "price": round(price, 6),
            "reason": reason,
        }

        async with JSON_ALERTS_FILE_LOCK:
            if JSON_ALERTS_FILE.exists():
                async with aiofiles.open(JSON_ALERTS_FILE, "r") as f:
                    try:
                        content = await f.read()
                        data = json.loads(content) if content else []
                    except json.JSONDecodeError:
                        data = []
            else:
                data = []

            data.append(alert)

            async with aiofiles.open(JSON_ALERTS_FILE, "w") as f:
                await f.write(json.dumps(data, indent=4))

        self.alerts[reason] = now_ms

        if reason == 'vwap':
            alert_price = self.vwap
        else:
            alert_price = self.ema_20

        self.logger.info(
            f"{self.symbol} ALERT | {reason.upper()}={alert_price} | price={price:.6f}"
        )
