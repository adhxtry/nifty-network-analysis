"""
Unit tests for niftynet.graph module.
"""

import pytest
import pandas as pd
import networkx as nx
from niftynet import graph


def test_compute_correlation_matrix():
    """Test correlation matrix computation."""
    # Create sample price data
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [2, 4, 6, 8, 10],  # Perfect positive correlation with A
        'C': [5, 4, 3, 2, 1]     # Perfect negative correlation with A
    })

    corr_matrix = graph.return_correlation_matrix(df, method='pearson')

    assert corr_matrix.shape == (3, 3)
    assert corr_matrix.loc['A', 'B'] > 0.99  # Almost perfect positive
    assert corr_matrix.loc['A', 'C'] < -0.99  # Almost perfect negative


def test_build_correlation_graph():
    """Test graph building from correlation matrix."""
    # Create sample correlation matrix
    corr_matrix = pd.DataFrame({
        'A': [1.0, 0.8, 0.2],
        'B': [0.8, 1.0, 0.3],
        'C': [0.2, 0.3, 1.0]
    }, index=['A', 'B', 'C'])

    # Build graph with threshold 0.5
    G = graph.build_correlation_graph(corr_matrix, threshold=0.5)

    assert G.number_of_nodes() == 3
    assert G.has_edge('A', 'B')  # Correlation 0.8 > 0.5
    assert not G.has_edge('A', 'C')  # Correlation 0.2 < 0.5


def test_filter_graph():
    """Test graph filtering by degree."""
    # Create a simple graph
    G = nx.Graph()
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('D', 'E')])

    # Filter with min_degree=2
    G_filtered = graph.filter_graph(G, min_degree=2)

    # Only B should remain (it has degree 2)
    assert 'B' in G_filtered.nodes()
    # C should be removed after B-C edge is gone when A is removed
    assert len(G_filtered.nodes()) >= 1


def test_get_graph_summary():
    """Test graph summary statistics."""
    G = nx.Graph()
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])

    summary = graph.get_graph_summary(G)

    assert summary['num_nodes'] == 3
    assert summary['num_edges'] == 3
    assert summary['is_connected'] == True
    assert 'avg_degree' in summary
