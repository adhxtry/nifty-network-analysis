"""
Unit tests for niftynet.visuals module.
"""

import pytest
import networkx as nx
import pandas as pd
from niftynet import visuals


def create_test_graph():
    """Create a simple test graph."""
    G = nx.Graph()
    G.add_edges_from([
        ('A', 'B'), ('A', 'C'), ('B', 'C')
    ])
    return G


def test_create_network_plot():
    """Test network plot creation."""
    G = create_test_graph()
    fig = visuals.create_network_plot(G, title="Test Network")

    assert fig is not None
    assert fig.layout.title.text == "Test Network"
    assert len(fig.data) == 2  # Edge trace and node trace


def test_create_network_plot_with_colors():
    """Test network plot with node colors."""
    G = create_test_graph()
    node_colors = {'A': 0.5, 'B': 0.8, 'C': 0.3}

    fig = visuals.create_network_plot(G, node_colors=node_colors)

    assert fig is not None
    # Node trace should have color information
    assert fig.data[1].marker.showscale == True


def test_create_correlation_heatmap():
    """Test correlation heatmap creation."""
    corr_matrix = pd.DataFrame({
        'A': [1.0, 0.8, 0.2],
        'B': [0.8, 1.0, 0.3],
        'C': [0.2, 0.3, 1.0]
    }, index=['A', 'B', 'C'])

    fig = visuals.create_correlation_heatmap(corr_matrix)

    assert fig is not None
    assert len(fig.data) == 1  # Heatmap trace


def test_create_centrality_bar_chart():
    """Test centrality bar chart creation."""
    centrality_data = pd.Series({
        'A': 0.5, 'B': 0.8, 'C': 0.3, 'D': 0.9, 'E': 0.7
    })

    fig = visuals.create_centrality_bar_chart(centrality_data, top_n=3)

    assert fig is not None
    # Should show only top 3
    assert len(fig.data[0].y) == 3


def test_create_degree_distribution():
    """Test degree distribution plot creation."""
    G = nx.erdos_renyi_graph(20, 0.2, seed=42)
    fig = visuals.create_degree_distribution(G)

    assert fig is not None
    assert len(fig.data) == 1  # Histogram trace
