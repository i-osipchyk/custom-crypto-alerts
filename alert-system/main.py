import asyncio
import logging
from pathlib import Path
from datetime import datetime, timezone

from modules.collector import run_all_symbols
from modules.symbol_loader import fetch_symbols


def setup_logger():
    Path("logs").mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(
                f"logs/run_{datetime.now(timezone.utc)}.log"
                .replace(" ", "_")
                .replace(":", "-")
            ),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("ALERT_SYSTEM")


def main():
    logger = setup_logger()
    symbols = fetch_symbols()
    logger.info(f"Loaded {len(symbols)} symbols")

    asyncio.run(run_all_symbols(symbols, logger))


if __name__ == "__main__":
    main()
