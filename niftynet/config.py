from pathlib import Path

config = {
    "cache_nifty_index": True,
    "cache_stock_data": True,
    "data_folder": Path.home() / ".niftynet" / "data",
    # Automatically prune stock data older than this many days whenever library is loaded
    "stock_cache_cutoff_days": 30,
}