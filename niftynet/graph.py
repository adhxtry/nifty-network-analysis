"""
Graph construction module for building correlation-based networks.

This module provides functions to create NetworkX graphs from stock price data
based on correlation relationships.
"""

from typing import Optional, Tuple
import networkx as nx
import numpy as np
import pandas as pd


def return_correlation_matrix(
    price_data: pd.DataFrame,
    method: str = "pearson",
    absolute: bool = True
) -> pd.DataFrame:
    """
    Compute correlation matrix from price data.
    First, computes the daily returns, then calculates the correlation matrix.

    The daily returns are calculated as `r_i(t) = log(P_i(t) / P_i(t-1))`
    where `P_i(t)` is the price of stock `i` at time `t`.

    Args:
        price_data: DataFrame with stock prices (tickers as columns)
        method: Correlation method - 'pearson', 'kendall', or 'spearman'
        absolute: Use absolute correlation values (default: True)

    Returns:
        Correlation matrix as DataFrame

    Raises:
        ValueError: If method is not recognized
    """
    valid_methods = ["pearson", "kendall", "spearman"]
    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}")

    # Log of returns
    log_returns = np.log(price_data / price_data.shift(1)).dropna(axis=0)

    corr = log_returns.corr(method=method)

    if absolute:
        return corr.abs()
    return corr


def build_correlation_graph(
    correlation_matrix: pd.DataFrame,
    threshold: float = 0.5
) -> nx.Graph:
    """
    Build a NetworkX graph from correlation matrix.

    Args:
        correlation_matrix: Correlation matrix as DataFrame
        threshold: Minimum correlation value for edge creation (default: 0.5)
        absolute: Use absolute correlation values (default: True)

    Returns:
        NetworkX Graph with nodes as tickers and edges weighted by correlation

    Raises:
        ValueError: If threshold is not between 0 and 1
    """
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")

    G = nx.Graph()

    G.add_nodes_from(correlation_matrix.index)

    # Add edges based on the threshold
    rows, cols = np.where(correlation_matrix.values >= threshold)
    mask = rows != cols  # Exclude self-loops
    rows, cols = rows[mask], cols[mask]
    rows, cols = correlation_matrix.index[rows], correlation_matrix.columns[cols]
    edges = zip(rows, cols)

    G.add_edges_from(edges)

    return G


def get_graph_summary(graph: nx.Graph) -> dict:
    """
    Get summary statistics of the graph.

    Args:
        graph: NetworkX graph to analyze

    Returns:
        Dictionary with graph statistics
    """
    summary = {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "is_connected": nx.is_connected(graph),
    }

    if graph.number_of_nodes() > 0:
        summary["avg_degree"] = sum(dict(graph.degree()).values()) / graph.number_of_nodes()

        if nx.is_connected(graph):
            summary["diameter"] = nx.diameter(graph)
            summary["avg_shortest_path"] = nx.average_shortest_path_length(graph)
        else:
            summary["num_components"] = nx.number_connected_components(graph)

    return summary


def degree_distribution(graph) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the degree distribution of a graph.

    Args:
        graph: NetworkX graph

    Returns:
        (degrees, counts): unique degree values and their corresponding frequencies
    """
    degrees = [degree for node, degree in graph.degree()]
    degree_counts = pd.Series(degrees).value_counts().sort_index()
    return degree_counts.index.values, degree_counts.values