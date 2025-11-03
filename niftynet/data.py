"""
Data fetching and storage module for stock market data.

This module provides functions to fetch historical stock data using yfinance
and store it in CSV format.
"""

from pathlib import Path
from typing import List, Optional
import pandas as pd
import yfinance as yf


def fetch_stock_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    column: str = "Close"
) -> pd.DataFrame:
    """
    Fetch historical stock data for multiple tickers.

    Args:
        tickers: List of stock ticker symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        column: Price column to extract (default: "Close")

    Returns:
        DataFrame with tickers as columns and dates as index

    Raises:
        ValueError: If no valid data is fetched
    """
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)

    if data.empty:
        raise ValueError("No data fetched. Check ticker symbols and date range.")

    # Handle multi-level columns for multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        prices = data[column]
    else:
        prices = data[[column]]

    # Remove any tickers with all NaN values
    prices = prices.dropna(axis=1, how='all')

    return prices


def save_to_csv(
    data: pd.DataFrame,
    output_dir: str = "data",
    filename: str = "stock_data.csv"
) -> Path:
    """
    Save DataFrame to CSV file in specified directory.

    Args:
        data: DataFrame to save
        output_dir: Directory to save the file (default: "data")
        filename: Name of the output file (default: "stock_data.csv")

    Returns:
        Path object pointing to the saved file

    Raises:
        IOError: If unable to write file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / filename
    data.to_csv(file_path)

    return file_path


def load_from_csv(file_path: str) -> pd.DataFrame:
    """
    Load stock data from CSV file.

    Args:
        file_path: Path to the CSV file

    Returns:
        DataFrame with loaded data

    Raises:
        FileNotFoundError: If file doesn't exist
        pd.errors.ParserError: If CSV is malformed
    """
    return pd.read_csv(file_path, index_col=0, parse_dates=True)


def fetch_and_save(
    tickers: List[str],
    start_date: str,
    end_date: str,
    output_dir: str = "data",
    filename: Optional[str] = None
) -> Path:
    """
    Fetch stock data and save it to CSV in one operation.

    Args:
        tickers: List of stock ticker symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        output_dir: Directory to save the file (default: "data")
        filename: Name of the output file (default: auto-generated)

    Returns:
        Path object pointing to the saved file
    """
    data = fetch_stock_data(tickers, start_date, end_date)

    if filename is None:
        filename = f"stock_data_{start_date}_to_{end_date}.csv"

    return save_to_csv(data, output_dir, filename)
