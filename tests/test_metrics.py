"""
Unit tests for niftynet.metrics module.
"""

import pytest
import networkx as nx
from niftynet import metrics


def create_test_graph():
    """Create a simple test graph."""
    G = nx.Graph()
    G.add_edges_from([
        ('A', 'B'), ('A', 'C'), ('B', 'C'),
        ('C', 'D'), ('D', 'E')
    ])
    return G


def test_compute_degree_centrality():
    """Test degree centrality computation."""
    G = create_test_graph()
    degree_cent = metrics.compute_degree_centrality(G)

    assert 'A' in degree_cent
    assert 'C' in degree_cent
    # C has highest degree (connected to A, B, D)
    assert degree_cent['C'] > degree_cent['E']


def test_compute_betweenness_centrality():
    """Test betweenness centrality computation."""
    G = create_test_graph()
    between_cent = metrics.compute_betweenness_centrality(G)

    assert 'D' in between_cent
    # D is a bridge between the triangle and E
    assert between_cent['D'] > 0


def test_compute_all_centralities():
    """Test computing all centrality measures."""
    G = create_test_graph()
    centralities_df = metrics.compute_all_centralities(G)

    assert 'degree' in centralities_df.columns
    assert 'betweenness' in centralities_df.columns
    assert 'closeness' in centralities_df.columns
    assert 'pagerank' in centralities_df.columns
    assert len(centralities_df) == G.number_of_nodes()


def test_get_top_nodes():
    """Test getting top nodes by centrality."""
    centrality_dict = {'A': 0.5, 'B': 0.8, 'C': 0.3, 'D': 0.9}
    top_nodes = metrics.get_top_nodes(centrality_dict, top_n=2)

    assert len(top_nodes) == 2
    assert top_nodes.index[0] == 'D'  # Highest value
    assert top_nodes.index[1] == 'B'  # Second highest


def test_compute_clustering_coefficient():
    """Test clustering coefficient computation."""
    G = nx.complete_graph(4)  # Fully connected graph
    clustering = metrics.compute_clustering_coefficient(G)

    # All nodes in complete graph have clustering coefficient of 1
    for node in clustering:
        assert clustering[node] == 1.0
