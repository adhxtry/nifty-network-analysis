"""
Network metrics computation module.

This module provides functions to compute various centrality measures and
other graph metrics for network analysis.
"""

from typing import Dict
import networkx as nx
import pandas as pd
import numpy as np


def log_linear_fitting(degrees: np.ndarray, counts: np.ndarray) -> tuple[callable, float]:
    """
    Fit a log-linear model to the degree distribution.

    Args:
        degrees: Array of degree values
        counts: Array of corresponding counts for each degree

    Returns:
        (model, r_squared): where model is a polynomial function and r_squared is the R² value
    """
    # Filter out degrees <= 1 to avoid log(0) and log(1) issues
    mask = degrees > 1
    degrees_filtered = degrees[mask]
    counts_filtered = counts[mask]

    if len(degrees_filtered) < 2:
        # Return a dummy polynomial and NaN error if not enough points
        return np.poly1d([0, 0]), np.nan

    log_degrees = np.log(degrees_filtered)
    log_counts = np.log(counts_filtered)
    coeffs = np.polyfit(log_degrees, log_counts, 1)
    model = np.poly1d(coeffs)

    # Calculate R² (coefficient of determination)
    y_pred = model(log_degrees)
    ss_res = np.sum((log_counts - y_pred) ** 2)  # Residual sum of squares
    ss_tot = np.sum((log_counts - np.mean(log_counts)) ** 2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)

    return model, r_squared


def compute_degree_centrality(graph: nx.Graph) -> Dict[str, float]:
    """
    Compute degree centrality for all nodes.

    Args:
        graph: NetworkX graph

    Returns:
        Dictionary mapping node names to degree centrality values
    """
    return nx.degree_centrality(graph)


def compute_betweenness_centrality(
    graph: nx.Graph,
    normalized: bool = True
) -> Dict[str, float]:
    """
    Compute betweenness centrality for all nodes.

    Args:
        graph: NetworkX graph
        normalized: Whether to normalize values (default: True)

    Returns:
        Dictionary mapping node names to betweenness centrality values
    """
    return nx.betweenness_centrality(graph, normalized=normalized)


def compute_closeness_centrality(
    graph: nx.Graph
) -> Dict[str, float]:
    """
    Compute closeness centrality for all nodes.

    Only computed for connected graphs or largest connected component.

    Args:
        graph: NetworkX graph

    Returns:
        Dictionary mapping node names to closeness centrality values
    """
    if nx.is_connected(graph):
        return nx.closeness_centrality(graph)
    else:
        # Compute for largest connected component
        largest_cc = max(nx.connected_components(graph), key=len)
        subgraph = graph.subgraph(largest_cc)
        return nx.closeness_centrality(subgraph)


def compute_eigenvector_centrality(
    graph: nx.Graph,
    max_iter: int = 100
) -> Dict[str, float]:
    """
    Compute eigenvector centrality for all nodes.

    Args:
        graph: NetworkX graph
        max_iter: Maximum number of iterations (default: 100)

    Returns:
        Dictionary mapping node names to eigenvector centrality values

    Raises:
        nx.PowerIterationFailedConvergence: If algorithm doesn't converge
    """
    try:
        return nx.eigenvector_centrality(graph, max_iter=max_iter)
    except nx.PowerIterationFailedConvergence:
        # Try with more iterations
        return nx.eigenvector_centrality(graph, max_iter=max_iter * 10)


def compute_pagerank(
    graph: nx.Graph,
    alpha: float = 0.85
) -> Dict[str, float]:
    """
    Compute PageRank for all nodes.

    Args:
        graph: NetworkX graph
        alpha: Damping parameter (default: 0.85)

    Returns:
        Dictionary mapping node names to PageRank values
    """
    return nx.pagerank(graph, alpha=alpha)


def compute_all_centralities(
    graph: nx.Graph
) -> pd.DataFrame:
    """
    Compute all centrality measures and return as DataFrame.

    Args:
        graph: NetworkX graph

    Returns:
        DataFrame with nodes as index and centrality measures as columns
    """
    metrics = {}

    metrics["degree"] = compute_degree_centrality(graph)
    metrics["betweenness"] = compute_betweenness_centrality(graph)
    metrics["closeness"] = compute_closeness_centrality(graph)

    try:
        metrics["eigenvector"] = compute_eigenvector_centrality(graph)
    except nx.PowerIterationFailedConvergence:
        metrics["eigenvector"] = {node: 0.0 for node in graph.nodes()}

    metrics["pagerank"] = compute_pagerank(graph)

    # Convert to DataFrame
    df = pd.DataFrame(metrics)
    df.index.name = "node"

    return df


def get_top_nodes(
    centrality_dict: Dict[str, float],
    top_n: int = 10
) -> pd.Series:
    """
    Get top N nodes by centrality value.

    Args:
        centrality_dict: Dictionary of centrality values
        top_n: Number of top nodes to return (default: 10)

    Returns:
        Series with top N nodes and their values, sorted descending
    """
    series = pd.Series(centrality_dict)
    return series.nlargest(top_n)


def compute_clustering_coefficient(graph: nx.Graph) -> Dict[str, float]:
    """
    Compute clustering coefficient for all nodes.

    Args:
        graph: NetworkX graph

    Returns:
        Dictionary mapping node names to clustering coefficient values
    """
    return nx.clustering(graph)


def compute_community_structure(
    graph: nx.Graph,
    algorithm: str = "louvain"
) -> Dict[str, int]:
    """
    Detect community structure in the graph.

    Args:
        graph: NetworkX graph
        algorithm: Community detection algorithm - 'louvain' or 'greedy'

    Returns:
        Dictionary mapping node names to community IDs

    Raises:
        ValueError: If invalid algorithm specified
    """
    if algorithm == "louvain":
        try:
            import community as community_louvain
            return community_louvain.best_partition(graph)
        except ImportError:
            # Fallback to greedy modularity
            algorithm = "greedy"

    if algorithm == "greedy":
        communities = nx.community.greedy_modularity_communities(graph)
        # Convert to dictionary
        node_to_community = {}
        for i, community in enumerate(communities):
            for node in community:
                node_to_community[node] = i
        return node_to_community
    else:
        raise ValueError("Algorithm must be 'louvain' or 'greedy'")
