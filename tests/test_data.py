"""
Unit tests for niftynet.data module.
"""

import pytest
import pandas as pd
from pathlib import Path
from niftynet import data


def test_fetch_stock_data():
    """Test basic stock data fetching functionality."""
    # This is a placeholder test
    # In real scenarios, you'd mock yfinance or use recorded responses
    pass


def test_save_to_csv(tmp_path):
    """Test CSV saving functionality."""
    # Create sample data
    df = pd.DataFrame({
        'AAPL': [100, 101, 102],
        'GOOGL': [200, 201, 202]
    }, index=pd.date_range('2023-01-01', periods=3))

    # Save to temp directory
    output_path = data.save_to_csv(df, output_dir=str(tmp_path), filename="test.csv")

    # Verify file exists
    assert output_path.exists()

    # Verify content
    loaded_df = pd.read_csv(output_path, index_col=0)
    assert loaded_df.shape == df.shape


def test_load_from_csv(tmp_path):
    """Test CSV loading functionality."""
    # Create and save sample data
    df = pd.DataFrame({
        'AAPL': [100, 101, 102],
        'GOOGL': [200, 201, 202]
    }, index=pd.date_range('2023-01-01', periods=3))

    file_path = tmp_path / "test.csv"
    df.to_csv(file_path)

    # Load and verify
    loaded_df = data.load_from_csv(str(file_path))
    assert loaded_df.shape[0] == 3
