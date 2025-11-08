"""
Data fetching and storage module for stock market data.

This module provides functions to fetch historical stock data using yfinance
and store it in CSV format.
"""

from .config import config

from pathlib import Path
import requests
from io import StringIO
import pandas as pd
import yfinance as yf
import time


def get_data_folder() -> Path:
    """
    Get the default data folder path.
    The library store in user's home directory under '.niftynet/data'.

    Returns:
        Path object pointing to the data directory
    """
    data_folder = config.get("data_folder", Path.home() / ".niftynet" / "data")
    data_folder.mkdir(parents=True, exist_ok=True)
    return data_folder


_TICKER_COLUMN = "Symbol"


def fetch_index(
    url: str = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    force_refresh: bool = False,
    ticker_column: str = "Symbol"
) -> pd.DataFrame:
    """
    Fetch the current list of companies index constituents.

    Args:
        url: URL to fetch the companies index list CSV
        force_refresh: If True, re-fetch the data even if cached (default: False)

    Returns:
        DataFrame with companies index constituents
    """
    force_refresh = force_refresh or not config.get("cache_nifty_index", True)

    data_folder = get_data_folder()
    index_file = data_folder / "index.csv"

    if index_file.exists() and not force_refresh:
        df = pd.read_csv(index_file)
    else:
        response = requests.get(
            url, timeout=10, allow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/csv,application/octet-stream",
            }
        )
        df = pd.read_csv(StringIO(response.text))
        # rename ticker_column column to _TICKER_COLUMN
        df.rename(columns={ticker_column: _TICKER_COLUMN}, inplace=True)
        df.to_csv(index_file, index=False)

    return df


def _fetch_stock_data_yfinance(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch the stock data using yfinance for all Nifty 500 tickers."""

    data_folder = get_data_folder() / "stock_data_cache"

    tickers = fetch_index()[_TICKER_COLUMN].dropna().unique().tolist()
    tickers = [ticker + ".NS" for ticker in tickers]

    data = yf.download(
        tickers, start=start_date, end=end_date, progress=False,
    )

    if data.empty:
        raise ValueError("No data fetched. Check ticker symbols and date range.")

    # Handle multi-level columns for multiple tickers.
    # save the cache to CSV before selecting column for each column name
    for col in data.columns.levels[0]:
        cache_file = data_folder / f"{start_date}-{end_date}-{col}.csv"
        data[col].to_csv(cache_file)

    # Remove any tickers with all NaN values
    return data


def fetch_stock_data(
    start_date: str,
    end_date: str,
    column: str = "Close"
) -> pd.DataFrame:
    """
    Fetch historical stock data for multiple tickers.
    Caches data in library's data folder.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        column: Price column to extract (default: "Close")

    Returns:
        DataFrame with tickers as columns and dates as index

    Raises:
        ValueError: If no valid data is fetched
    """

    data_folder = get_data_folder() / "stock_data_cache"

    if not data_folder.exists():
        data_folder.mkdir(exist_ok=True)

    filename = f"{start_date}-{end_date}-{column}.csv"

    cache_file = data_folder / filename

    if cache_file.exists() and config.get("cache_stock_data", True):
        prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return prices

    data = _fetch_stock_data_yfinance(start_date, end_date)
    if column not in data.columns.levels[0]:
        raise ValueError(f"Column '{column}' not found in yfinance fetched data.")

    return data[column]


def purge_stock_cache(cutoff_days: int = 30):
    """
    Prune cached stock data files older than `cutoff_days`.
    """

    data_folder: Path = get_data_folder()

    # remove files in index first
    for file in data_folder.iterdir():
        if file.is_file() and file.name.endswith(".csv"):
            file.unlink()

    # remove stock data cache
    data_folder = data_folder / "stock_data_cache"
    if not data_folder.exists():
        return

    now = time.time()
    cutoff = now - (cutoff_days * 86400)  # 30 days in seconds

    for file in data_folder.iterdir():
        if file.is_file():
            file_mtime = file.stat().st_mtime
            if file_mtime < cutoff:
                file.unlink()


purge_stock_cache(config.get("stock_cache_cutoff_days", 30))