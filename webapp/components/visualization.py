"""
Visualization components for the Dash web application.
"""

from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import networkx as nx
import random


def create_visualization_placeholder():
    """
    Create a placeholder visualization with a random NetworkX graph.

    Returns:
        Dash Bootstrap Components layout with placeholder graph
    """
    # Generate a random graph for placeholder
    random.seed(42)
    G = nx.erdos_renyi_graph(20, 0.15, seed=42)

    # Create positions using spring layout
    pos = nx.spring_layout(G, seed=42)

    # Create edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
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
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color='lightblue',
            size=15,
            line=dict(width=2, color='white')
        ),
        text=[f"Node {node}" for node in G.nodes()]
    )

    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text="Sample Network (Configure inputs above to analyze real data)",
                font=dict(size=16)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
    )

    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H3("Network Visualization Preview", className="mt-4 mb-3"),
                html.P(
                    "This is a placeholder visualization. Configure the inputs above "
                    "and click 'Analyze Network' to see actual stock correlation data.",
                    className="text-muted"
                ),
                dcc.Graph(figure=fig)
            ])
        ])
    ])
