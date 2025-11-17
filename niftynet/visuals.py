"""
Visualization module for creating interactive network plots.

This module provides functions to generate interactive visualizations
using Plotly for network graphs and other data.
"""

import numpy as np
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots

def _sort_nodes_by_centrality(
    graph: nx.Graph,
    centrality_map: pd.Series
) -> list:
    """Sort nodes by centrality score in descending order."""
    return sorted(
        list(graph.nodes()),
        key=lambda n: centrality_map.get(n, 0),
        reverse=True
    )


def _calculate_group_sizes(total_nodes: int) -> tuple:
    """Calculate sizes for inner, middle, and outer groups."""
    n_inner = min(5, total_nodes)
    n_remaining = total_nodes - n_inner
    n_middle = int(np.ceil(n_remaining * 0.25)) if n_remaining > 0 else 0
    n_outer = n_remaining - n_middle
    return n_inner, n_middle, n_outer


def _split_into_groups(sorted_nodes: list, n_inner: int, n_middle: int) -> tuple:
    """Split sorted nodes into inner, middle, and outer groups."""
    inner_group = sorted_nodes[:n_inner]
    remaining_nodes = sorted_nodes[n_inner:]
    middle_group = remaining_nodes[:n_middle]
    outer_group = remaining_nodes[n_middle:]
    return inner_group, middle_group, outer_group


def _place_nodes_in_circle(nodes: list, radius: float) -> dict[str, np.ndarray]:
    """Place nodes in a circular layout at given radius."""
    pos = {}
    count = len(nodes)
    if count == 0:
        return pos

    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    for i, node in enumerate(nodes):
        theta = angles[i]
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        pos[node] = np.array([x, y])
    return pos


def _randomize_group_order(group: list, seed: int) -> list:
    """Randomize the order of nodes in a group deterministically."""
    rng = np.random.RandomState(seed)
    shuffled = group.copy()
    rng.shuffle(shuffled)
    return shuffled


def get_stratified_circular_layout(
    graph: nx.Graph,
    centrality_map: pd.Series,
    scale: float = 1.0
) -> dict[str, np.ndarray]:
    """
    Generates positions for a 3-ring concentric layout based on node ranking.

    Logic:
    1. Inner Circle: Top 5 nodes
    2. Middle Circle: 25% of the remaining nodes
    3. Outer Circle: The rest

    Args:
        graph: The NetworkX graph
        centrality_map: Dictionary of {node: centrality_score}
        scale: Scale factor for the graph coordinates (default 1.0)

    Returns:
        Dictionary of {node: array([x, y])}
    """
    sorted_nodes = _sort_nodes_by_centrality(graph, centrality_map)
    total_nodes = len(sorted_nodes)

    n_inner, n_middle, n_outer = _calculate_group_sizes(total_nodes)
    inner_group, middle_group, outer_group = _split_into_groups(sorted_nodes, n_inner, n_middle)

    # Randomize middle and outer groups deterministically
    middle_group = _randomize_group_order(middle_group, seed=9104)
    outer_group = _randomize_group_order(outer_group, seed=9104)

    radii = [1.0 * scale, 2.2 * scale, 3.5 * scale]

    pos = {}
    pos.update(_place_nodes_in_circle(inner_group, radii[0]))
    pos.update(_place_nodes_in_circle(middle_group, radii[1]))
    pos.update(_place_nodes_in_circle(outer_group, radii[2]))

    return pos


def _compute_layout_positions(
    graph: nx.Graph,
    layout: str,
    node_size_metric: pd.Series | None = None
) -> dict:
    """Compute node positions based on layout algorithm."""
    if layout == "stratified":
        if node_size_metric is None:
            raise ValueError("node_size_metric must be provided for stratified layout")
        return get_stratified_circular_layout(graph, node_size_metric, scale=1.0)
    elif layout == "spring":
        return nx.spring_layout(graph, seed=42)
    elif layout == "circular":
        return nx.circular_layout(graph)
    elif layout == "kamada_kawai":
        return nx.kamada_kawai_layout(graph)
    else:
        raise ValueError("Layout must be 'spring', 'circular', 'kamada_kawai', or 'stratified'")


def _calculate_node_sizes(
    nodes: list,
    node_size_metric: pd.Series | None = None
) -> list:
    """Calculate node sizes based on metric values."""
    if node_size_metric is None:
        return [15] * len(nodes)

    nsm_values = node_size_metric.values
    if nsm_values.max() > nsm_values.min():
        normalized_nsm = (nsm_values - nsm_values.min()) / (nsm_values.max() - nsm_values.min())
    else:
        normalized_nsm = np.zeros_like(nsm_values)

    normalized_nsm_series = pd.Series(normalized_nsm, index=node_size_metric.index)
    return [10 + 40 * normalized_nsm_series.get(n, 0) for n in nodes]


def _build_node_data(
    graph: nx.Graph,
    pos: dict,
    node_size_metric: pd.Series | None = None
) -> tuple:
    """Build node coordinates, text, and sizes."""
    nodes = list(graph.nodes())
    deg_map = dict(graph.degree(nodes))

    node_x = [pos[n][0] for n in nodes]
    node_y = [pos[n][1] for n in nodes]
    node_text = [f"{n}<br>Degree: {deg_map[n]}" for n in nodes]
    node_sizes = _calculate_node_sizes(nodes, node_size_metric)

    return nodes, node_x, node_y, node_text, node_sizes


def _build_edge_coordinates(graph: nx.Graph, pos: dict) -> tuple:
    """Build edge coordinates for plotting."""
    edges = list(graph.edges())
    if not edges:
        return [], []

    edge_x = [coord for e in edges for coord in (pos[e[0]][0], pos[e[1]][0], None)]
    edge_y = [coord for e in edges for coord in (pos[e[0]][1], pos[e[1]][1], None)]
    return edge_x, edge_y


def _create_edge_trace(edge_x: list, edge_y: list) -> go.Scatter:
    """Create edge trace for network plot."""
    return go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.2, color="#888"),
        opacity=0.5,
        hoverinfo="none",
        mode="lines",
        name='Edges'
    )


def _create_node_trace(
    node_x: list,
    node_y: list,
    node_text: list,
    node_sizes: list,
    nodes: list,
    add_labels: bool = False,
    labels: list | None = None,
    node_size_metric: pd.Series | None = None
) -> go.Scatter:
    """Create node trace for network plot."""
    if add_labels and labels is None:
        labels = nodes

    # Get color values based on node_size_metric with proper min/max scaling
    if node_size_metric is not None:
        color_values = [node_size_metric.get(n, 0) for n in nodes]
        cmin = node_size_metric.min()
        cmax = node_size_metric.max()
    else:
        color_values = [1] * len(nodes)
        cmin = 0
        cmax = 1

    marker_dict = dict(
        color=color_values,
        colorscale="Viridis",
        showscale=True,
        cmin=cmin,
        cmax=cmax,
        colorbar=dict(
            thickness=15,
            title="Centrality",
            xanchor="left"
        ),
        size=node_sizes,
        line=dict(width=0.5, color="white")
    )

    return go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers" if not add_labels else "markers+text",
        text=labels if add_labels else None,
        textposition="top center" if add_labels else None,
        textfont={
            "size": 10,
            "color": "white",
            "shadow": "#000000 1px 0 10px"
        },
        hoverinfo="text",
        hovertext=node_text,
        marker=marker_dict,
        name='Nodes'
    )


def _get_network_layout_config(title: str) -> dict:
    """Get layout configuration for network plot."""
    return dict(
        title=dict(text=title, font=dict(size=14)),
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis_scaleanchor="x",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )


def create_network_plot(
    graph: nx.Graph,
    node_size_metric: pd.Series | None = None,
    layout: str = "spring",
    title: str = "Network"
) -> go.Figure:
    """
    Create an interactive network visualization using Plotly.

    Args:
        graph: NetworkX graph to visualize
        node_size_metric: Dictionary mapping nodes to size values (default: None)
        layout: Layout algorithm - 'spring', 'circular', 'kamada_kawai', 'stratified' (default: 'spring')
        title: Plot title (default: "Network Graph")

    Returns:
        Tuple of (traces, layout_config) for use in subplots

    Raises:
        ValueError: If invalid layout specified
    """
    pos = _compute_layout_positions(graph, layout, node_size_metric)
    nodes, node_x, node_y, node_text, node_sizes = _build_node_data(graph, pos, node_size_metric)
    edge_x, edge_y = _build_edge_coordinates(graph, pos)

    edge_trace = _create_edge_trace(edge_x, edge_y)

    # Add labels for the first five nodes
    labels = [None] * len(nodes)
    for n in node_size_metric.nlargest(5).index:
        if n in nodes:
            idx = nodes.index(n)
            labels[idx] = n

    node_trace = _create_node_trace(node_x, node_y, node_text, node_sizes, nodes, True, labels, node_size_metric)
    layout_config = _get_network_layout_config(title)

    return [edge_trace, node_trace], layout_config


def _create_bar_trace(top_data: pd.Series) -> go.Bar:
    """Create bar chart trace for centrality rankings."""
    return go.Bar(
        x=top_data.values,
        y=top_data.index,
        orientation='h',
        marker=dict(
            color=top_data.values,
            colorscale='Viridis'
        ),
        name="Centrality"
    )


def _add_bar_chart_subplot(fig: go.Figure, top_data: pd.Series) -> None:
    """Add bar chart subplot to figure."""
    bar_trace = _create_bar_trace(top_data)
    fig.add_trace(bar_trace, row=1, col=1)
    fig.update_xaxes(title_text="Centrality Value", row=1, col=1)
    fig.update_yaxes(title_text="Node", automargin=True, row=1, col=1)


def _add_network_subplot(
    fig: go.Figure,
    graph: nx.Graph,
    centrality_data: pd.Series
) -> None:
    """Add network visualization subplot to figure."""
    network_traces, network_layout_updates = create_network_plot(
        graph=graph,
        node_size_metric=centrality_data,
        layout="stratified",
        title=""
    )
    if network_traces:
        rows = [1] * len(network_traces)
        cols = [2] * len(network_traces)
        fig.add_traces(network_traces, rows=rows, cols=cols)

    fig.update_xaxes(network_layout_updates['xaxis'], row=1, col=2)
    fig.update_yaxes(network_layout_updates['yaxis'], row=1, col=2)


def create_centrality_chart(
    graph: nx.Graph,
    centrality_data: pd.Series,
    title: str = "Centrality Analysis",
    top_n: int = 20
) -> go.Figure:
    """
    Create a combined plot: a bar chart for centrality rankings and a network
    plot where nodes are sized by the centrality metric.

    Args:
        graph: NetworkX graph for the visualization.
        centrality_data: Series with node names as index and values.
        title: Overall plot title (default: "Centrality Analysis").
        top_n: Number of top nodes to display in the bar chart (default: 20).

    Returns:
        Plotly Figure object with subplots.
    """
    top_data = centrality_data.nlargest(top_n).sort_values()
    bar_chart_title = f"Top {top_n} Centrality Ranking"

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.5, 0.5],
        subplot_titles=(bar_chart_title, "Network Visualization (Sized by Centrality)"),
        horizontal_spacing=0.05,
    )

    _add_bar_chart_subplot(fig, top_data)
    _add_network_subplot(fig, graph, centrality_data)

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20)),
        height=max(600, top_n * 25),
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="closest",
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
