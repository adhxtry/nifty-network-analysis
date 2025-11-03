"""
Dash web application for Nifty Stock Network Analysis.

This app allows users to input a CSV URL with stock tickers, select date ranges,
and visualize network relationships between stocks.
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta
import requests
from io import StringIO

from components.inputs import create_input_section
from components.visualization import create_visualization_placeholder
from niftynet import data as nifty_data, graph as nifty_graph, visuals as nifty_visuals

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# Disable dev tools that can cause chunk loading issues
app.config.suppress_callback_exceptions = True

app.title = "Nifty Network Analysis"

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Nifty Stock Network Analysis", className="text-center mb-4 mt-4"),
            html.P(
                "Analyze network relationships between Nifty500 companies using stock market data.",
                className="text-center text-muted mb-4"
            )
        ])
    ]),

    # Input section
    create_input_section(),

    # Loading spinner
    dcc.Loading(
        id="loading",
        type="default",
        children=[
            # Status message
            html.Div(id="status-message", className="mb-3"),

            # Visualization section
            html.Div(id="visualization-container")
        ]
    )
], fluid=True)


@app.callback(
    [Output("visualization-container", "children"),
     Output("status-message", "children")],
    [Input("analyze-button", "n_clicks")],
    [State("csv-url-input", "value"),
     State("start-date-picker", "date"),
     State("end-date-picker", "date"),
     State("correlation-threshold", "value")]
)
def update_visualization(n_clicks, csv_url, start_date, end_date, threshold):
    """
    Update visualization based on user inputs.

    Args:
        n_clicks: Number of button clicks
        csv_url: URL to CSV file with tickers
        start_date: Start date for data fetch
        end_date: End date for data fetch
        threshold: Correlation threshold for graph edges

    Returns:
        Tuple of (visualization components, status message)
    """
    if n_clicks is None or n_clicks == 0:
        # Show placeholder on initial load
        return create_visualization_placeholder(), ""

    # Validate inputs
    if not csv_url:
        return html.Div(), dbc.Alert("Please provide a CSV URL.", color="warning")

    if not start_date or not end_date:
        return html.Div(), dbc.Alert("Please select both start and end dates.", color="warning")

    try:
        # Fetch tickers from CSV URL
        response = requests.get(csv_url, timeout=10)
        response.raise_for_status()

        # Parse CSV
        csv_data = pd.read_csv(StringIO(response.text))

        # Try to find ticker column (common names)
        ticker_col = None
        possible_cols = ['Symbol', 'symbol', 'Ticker', 'ticker', 'SYMBOL', 'TICKER']
        for col in possible_cols:
            if col in csv_data.columns:
                ticker_col = col
                break

        if ticker_col is None:
            # Use first column if no standard column found
            ticker_col = csv_data.columns[0]

        # Extract tickers and add .NS suffix for Indian stocks
        tickers = csv_data[ticker_col].dropna().unique().tolist()
        tickers = [f"{ticker}.NS" if not ticker.endswith('.NS') else ticker
                   for ticker in tickers]

        # Limit to first 50 tickers for performance
        if len(tickers) > 50:
            tickers = tickers[:50]
            status_msg = f"Limited to first 50 tickers for performance. Total available: {len(csv_data[ticker_col].dropna())}"
        else:
            status_msg = f"Processing {len(tickers)} tickers..."

        # Fetch stock data
        price_data = nifty_data.fetch_stock_data(tickers, start_date, end_date)

        # Save to CSV
        output_path = nifty_data.save_to_csv(price_data, output_dir="data",
                                             filename=f"stock_data_{start_date}_to_{end_date}.csv")

        # Build correlation graph
        graph, corr_matrix = nifty_graph.build_graph_from_prices(
            price_data,
            threshold=threshold/100.0,  # Convert from percentage
            method="pearson"
        )

        # Filter isolated nodes
        graph = nifty_graph.filter_graph(graph, min_degree=1)

        if graph.number_of_nodes() == 0:
            return html.Div(), dbc.Alert(
                f"No connections found with threshold {threshold}%. Try lowering the threshold.",
                color="warning"
            )

        # Create visualizations
        network_fig = nifty_visuals.create_network_plot(
            graph,
            title=f"Stock Correlation Network (threshold: {threshold}%)"
        )

        # Create layout with visualizations
        viz_layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Network Visualization", className="mt-4 mb-3"),
                    dcc.Graph(figure=network_fig)
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.H4("Graph Statistics", className="mt-4 mb-3"),
                    html.Div([
                        html.P(f"Number of Nodes: {graph.number_of_nodes()}"),
                        html.P(f"Number of Edges: {graph.number_of_edges()}"),
                        html.P(f"Network Density: {nx.density(graph):.4f}"),
                        html.P(f"Data saved to: {output_path}")
                    ])
                ])
            ])
        ])

        success_msg = dbc.Alert(
            f"✓ Successfully analyzed {len(tickers)} stocks from {start_date} to {end_date}",
            color="success"
        )

        return viz_layout, success_msg

    except requests.exceptions.RequestException as e:
        return html.Div(), dbc.Alert(f"Error fetching CSV: {str(e)}", color="danger")
    except Exception as e:
        return html.Div(), dbc.Alert(f"Error: {str(e)}", color="danger")


if __name__ == "__main__":
    # Run with use_reloader=False to avoid chunk loading issues in development
    app.run(debug=True, host="127.0.0.1", port=8050, use_reloader=False)
