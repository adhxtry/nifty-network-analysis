"""
Visualization module for creating interactive network plots.

This module provides functions to generate interactive visualizations
using Plotly for network graphs and other data.
"""

from typing import Dict, Optional
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import pandas as pd


def create_network_plot(
    graph: nx.Graph,
    node_colors: Optional[Dict[str, float]] = None,
    node_size_metric: Optional[Dict[str, float]] = None,
    layout: str = "spring",
    title: str = "Network Graph"
) -> go.Figure:
    """
    Create an interactive network visualization using Plotly.

    Args:
        graph: NetworkX graph to visualize
        node_colors: Dictionary mapping nodes to color values (default: None)
        node_size_metric: Dictionary mapping nodes to size values (default: None)
        layout: Layout algorithm - 'spring', 'circular', 'kamada_kawai' (default: 'spring')
        title: Plot title (default: "Network Graph")

    Returns:
        Plotly Figure object

    Raises:
        ValueError: If invalid layout specified
    """
    # Compute layout positions
    if layout == "spring":
        pos = nx.spring_layout(graph, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(graph)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(graph)
    else:
        raise ValueError("Layout must be 'spring', 'circular', or 'kamada_kawai'")

    # Create edge trace
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Create node trace
    node_x = []
    node_y = []
    node_text = []

    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}<br>Degree: {graph.degree(node)}")

    # Determine node colors
    if node_colors is not None:
        node_color_values = [node_colors.get(node, 0) for node in graph.nodes()]
        colorscale = 'Viridis'
        showscale = True
    else:
        node_color_values = '#1f77b4'
        colorscale = None
        showscale = False

    # Determine node sizes
    if node_size_metric is not None:
        node_sizes = [20 + 30 * node_size_metric.get(node, 0) for node in graph.nodes()]
    else:
        node_sizes = 20

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(
            showscale=showscale,
            colorscale=colorscale,
            color=node_color_values,
            size=node_sizes,
            colorbar=dict(
                thickness=15,
                title="Value",
                xanchor='left'
            ),
            line=dict(width=2, color='white')
        )
    )

    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text=title, font=dict(size=16)),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
    )

    return fig


def create_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    title: str = "Correlation Heatmap"
) -> go.Figure:
    """
    Create an interactive correlation heatmap.

    Args:
        correlation_matrix: Correlation matrix as DataFrame
        title: Plot title (default: "Correlation Heatmap")

    Returns:
        Plotly Figure object
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 8},
            colorbar=dict(title="Correlation")
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Ticker",
        yaxis_title="Ticker",
        height=600
    )

    return fig


def create_centrality_bar_chart(
    centrality_data: pd.Series,
    title: str = "Centrality Ranking",
    top_n: int = 20
) -> go.Figure:
    """
    Create a bar chart for centrality rankings.

    Args:
        centrality_data: Series with node names as index and values
        title: Plot title (default: "Centrality Ranking")
        top_n: Number of top nodes to display (default: 20)

    Returns:
        Plotly Figure object
    """
    top_data = centrality_data.nlargest(top_n).sort_values()

    fig = go.Figure(
        data=go.Bar(
            x=top_data.values,
            y=top_data.index,
            orientation='h',
            marker=dict(
                color=top_data.values,
                colorscale='Viridis'
            )
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Centrality Value",
        yaxis_title="Node",
        height=max(400, top_n * 20),
        showlegend=False
    )

    return fig


def create_degree_distribution(
    graph: nx.Graph,
    title: str = "Degree Distribution"
) -> go.Figure:
    """
    Create a histogram of degree distribution.

    Args:
        graph: NetworkX graph
        title: Plot title (default: "Degree Distribution")

    Returns:
        Plotly Figure object
    """
    degrees = [graph.degree(node) for node in graph.nodes()]

    fig = go.Figure(
        data=go.Histogram(
            x=degrees,
            nbinsx=20,
            marker=dict(color='steelblue')
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Degree",
        yaxis_title="Count",
        height=400
    )

    return fig


def plot_degree_distribution(
    degrees: np.ndarray, counts: np.ndarray,
    model: callable,
    fig_grid: go.Figure, row: int, col: int
) -> go.Figure:
    """
    Plot the degree distribution on a log-log scale.
    Also plots the fitted power-law line, with error bands.

    Adds the plot to the provided `fig_grid` figure.
    """
    # Add data points to the subplot
    fig_grid.add_trace(
        go.Scatter(x=degrees, y=counts, mode='markers', name='Data', showlegend=False),
        row=row,
        col=col
    )

    # Plot the log-linear model to the data
    x_fit = np.linspace(degrees[degrees > 0].min(), degrees.max(), 3, endpoint=True)
    y_fit = np.exp(model(np.log(x_fit)))

    # Add fitted line to the subplot
    fig_grid.add_trace(
        go.Scatter(x=x_fit, y=y_fit, mode='lines', name='Fitted Power Law', showlegend=False),
        row=row,
        col=col
    )
    return fig_grid
