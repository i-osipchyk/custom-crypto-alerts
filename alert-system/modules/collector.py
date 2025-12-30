import json
import asyncio
import aiohttp
import websockets
from datetime import datetime, timedelta, timezone

from modules.config import WS_RECONNECT_DELAY
from modules.symbol import Symbol


async def collect_symbol_data(symbol: str, tracker: Symbol, logger):
    url = (
        "wss://fstream.binance.com/stream"
        f"?streams={symbol}@kline_5m"
    )

    while True:
        try:
            async with websockets.connect(url) as ws:
                logger.info(f"{symbol} 5m WS connected")

                async for msg in ws:
                    data = json.loads(msg)
                    await tracker.update_from_kline(data["data"]["k"])

        except Exception as e:
            logger.warning(f"{symbol} WS error: {e}")
            await asyncio.sleep(WS_RECONNECT_DELAY)


async def run_all_symbols(symbols, logger):
    async with aiohttp.ClientSession() as session:
        tasks = []

        for sym in symbols:
            tracker = Symbol(sym, logger)

            await tracker.init_df(session)

            logger.info(f"{sym}: init complete, starting WS")

            task = asyncio.create_task(
                collect_symbol_data(sym, tracker, logger)
            )
            tasks.append(task)

        await asyncio.gather(*tasks)

