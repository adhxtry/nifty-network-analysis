"""
NiftyNet: Network analysis and visualization for Nifty companies.

A Python library for analyzing stock market relationships using network science.
"""

try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("nifty-network-analysis")
    except PackageNotFoundError:
        __version__ = "0.1.0"  # Fallback for development
except ImportError:
    __version__ = "0.1.0"  # Fallback for older Python versions

from . import data, graph, metrics, visuals

__all__ = ["__version__", "data", "graph", "metrics", "visuals"]