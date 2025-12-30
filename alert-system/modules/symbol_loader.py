import requests

BINANCE_FUTURES_REST = "https://fapi.binance.com"


def fetch_symbols(quote_asset: str = "USDT") -> list[str]:
    url = f"{BINANCE_FUTURES_REST}/fapi/v1/exchangeInfo"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()

    data = resp.json()

    return [
        s["symbol"].lower()
        for s in data["symbols"]
        if s["status"] == "TRADING"
        and s["contractType"] == "PERPETUAL"
        and s["quoteAsset"] == quote_asset
    ]
