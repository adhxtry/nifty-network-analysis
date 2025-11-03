"""
Graph construction module for building correlation-based networks.

This module provides functions to create NetworkX graphs from stock price data
based on correlation relationships.
"""

from typing import Optional, Tuple
import networkx as nx
import numpy as np
import pandas as pd


def compute_correlation_matrix(
    price_data: pd.DataFrame,
    method: str = "pearson"
) -> pd.DataFrame:
    """
    Compute correlation matrix from price data.

    Args:
        price_data: DataFrame with stock prices (tickers as columns)
        method: Correlation method - 'pearson', 'kendall', or 'spearman'

    Returns:
        Correlation matrix as DataFrame

    Raises:
        ValueError: If invalid correlation method specified
    """
    valid_methods = ["pearson", "kendall", "spearman"]
    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}")

    return price_data.corr(method=method)


def build_correlation_graph(
    correlation_matrix: pd.DataFrame,
    threshold: float = 0.5,
    absolute: bool = True
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

    # Add all nodes
    tickers = correlation_matrix.columns.tolist()
    G.add_nodes_from(tickers)

    # Add edges based on correlation threshold
    for i, ticker1 in enumerate(tickers):
        for j, ticker2 in enumerate(tickers):
            if i < j:  # Avoid duplicate edges and self-loops
                corr_value = correlation_matrix.iloc[i, j]

                # Use absolute value if specified
                corr_to_compare = abs(corr_value) if absolute else corr_value

                if corr_to_compare >= threshold:
                    G.add_edge(ticker1, ticker2, weight=corr_value)

    return G


def build_graph_from_prices(
    price_data: pd.DataFrame,
    threshold: float = 0.5,
    method: str = "pearson",
    absolute: bool = True
) -> Tuple[nx.Graph, pd.DataFrame]:
    """
    Build correlation graph directly from price data.

    Args:
        price_data: DataFrame with stock prices (tickers as columns)
        threshold: Minimum correlation value for edge creation (default: 0.5)
        method: Correlation method - 'pearson', 'kendall', or 'spearman'
        absolute: Use absolute correlation values (default: True)

    Returns:
        Tuple of (NetworkX Graph, correlation matrix)
    """
    corr_matrix = compute_correlation_matrix(price_data, method)
    graph = build_correlation_graph(corr_matrix, threshold, absolute)

    return graph, corr_matrix


def filter_graph(
    graph: nx.Graph,
    min_degree: int = 1
) -> nx.Graph:
    """
    Filter graph by removing isolated nodes or low-degree nodes.

    Args:
        graph: NetworkX graph to filter
        min_degree: Minimum degree for nodes to keep (default: 1)

    Returns:
        Filtered NetworkX graph
    """
    G_filtered = graph.copy()

    # Remove nodes with degree less than threshold
    nodes_to_remove = [node for node, degree in dict(G_filtered.degree()).items()
                       if degree < min_degree]
    G_filtered.remove_nodes_from(nodes_to_remove)

    return G_filtered


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
