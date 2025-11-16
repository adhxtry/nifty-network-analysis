"""
Nifty Network Analysis Web Application

A Dash-based web application for analyzing stock market networks using correlation-based graphs.
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO, BytesIO
import pandas as pd
import numpy as np
import networkx as nx
import json
import base64
from datetime import datetime
import zipfile

import niftynet as nn
from config import *

# Initialize Dash app with Bootstrap theme
app = dash.Dash(
    "Nifty Network Analysis",
    external_stylesheets=[getattr(dbc.themes, BOOTSTRAP_THEME)],
    suppress_callback_exceptions=True
)

app.title = "Nifty Network Analysis"

# ============================================================================
# LAYOUT COMPONENTS
# ============================================================================

def create_header():
    """Create the header section."""
    return dbc.Container([
        html.H1("📊 Nifty Network Analysis", className="text-center my-4"),
        html.P(
            "Analyze stock market networks using correlation-based graphs. "
            "Explore scale-free characteristics and identify influential stocks.",
            className="text-center text-muted mb-4"
        ),
        html.Hr()
    ], fluid=True)


def create_input_section():
    """Create the input parameters section."""
    return dbc.Container([
        html.H3("1️⃣ Data Configuration", className="mb-3"),
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Index URL:", html_for="index-url"),
                        dbc.Input(
                            id="index-url",
                            type="text",
                            value=DEFAULT_INDEX_URL,
                            placeholder="Enter NSE index CSV URL"
                        ),
                    ], md=10),
                    dbc.Col([
                        dbc.Label("Force Fetch:", html_for="force-fetch"),
                        dbc.Checkbox(
                            id="force-fetch",
                            value=False,
                            className="mt-2"
                        ),
                    ], md=2),
                ], className="mb-3"),

                dbc.Row([
                    dbc.Col([
                        dbc.Label("Start Date:", html_for="start-date"),
                        dbc.Input(
                            id="start-date",
                            type="date",
                            value=DEFAULT_START_DATE
                        ),
                    ], md=6),
                    dbc.Col([
                        dbc.Label("End Date:", html_for="end-date"),
                        dbc.Input(
                            id="end-date",
                            type="date",
                            value=DEFAULT_END_DATE
                        ),
                    ], md=6),
                ], className="mb-3"),

                dbc.Row([
                    dbc.Col([
                        dbc.Label("Threshold Range:", html_for="threshold-range"),
                        html.Div([
                            dcc.RangeSlider(
                                id="threshold-range",
                                min=0.0,
                                max=1.0,
                                step=DEFAULT_THRESHOLD_STEP,
                                value=[DEFAULT_THRESHOLD_MIN, DEFAULT_THRESHOLD_MAX],
                                marks={i/10: f'{i/10:.1f}' for i in range(0, 11, 2)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], className="mt-2")
                    ], md=12),
                ]),

                dbc.Button(
                    "🔍 Fetch Data & Generate Threshold Analysis",
                    id="fetch-button",
                    color="primary",
                    size="lg",
                    className="w-100 mt-4"
                ),
            ])
        ], className="shadow-sm")
    ], fluid=True, className="mb-5")


def create_threshold_analysis_section():
    """Create the threshold analysis results section."""
    return dbc.Container([
        html.Div(id="threshold-section", children=[
            dbc.Row([
                dbc.Col(html.H3("2️⃣ Power Law Analysis (Multiple Thresholds)", className="mb-3"), width=6),
                dbc.Col([
                    dbc.Button(
                        "📊 Download All (ZIP)",
                        id="download-threshold-btn",
                        color="info",
                        size="sm",
                        disabled=True,
                        n_clicks=0
                    ),
                    dcc.Download(id="download-threshold"),
                    html.Small("Downloads plot + stats as ZIP", className="text-muted d-block text-end mt-1")
                ], width=6)
            ]),
            dbc.Card([
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-threshold",
                        type="default",
                        children=html.Div(id="threshold-results")
                    )
                ])
            ], className="shadow-sm")
        ], style={"display": "none"})
    ], fluid=True, className="mb-5")


def create_detailed_analysis_section():
    """Create the detailed network analysis section."""
    return dbc.Container([
        html.Div(id="detailed-section", children=[
            dbc.Row([
                dbc.Col(html.H3("3️⃣ Detailed Network Analysis", className="mb-3"), width=6),
                dbc.Col([
                    dbc.Button(
                        "📥 Download All (ZIP)",
                        id="download-analysis-btn",
                        color="info",
                        size="sm",
                        className="float-end",
                        disabled=True,
                        n_clicks=0
                    ),
                    dcc.Download(id="download-analysis"),
                    html.Small("Downloads plots + report as ZIP", className="text-muted d-block text-end mt-1")
                ], width=6)
            ]),
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Select Threshold for Analysis:", html_for="analysis-threshold"),
                            dcc.Slider(
                                id="analysis-threshold",
                                min=0.0,
                                max=1.0,
                                step=DEFAULT_THRESHOLD_STEP,
                                value=DEFAULT_ANALYSIS_THRESHOLD,
                                marks={i/10: f'{i/10:.1f}' for i in range(0, 11, 2)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], md=10),
                        dbc.Col([
                            dbc.Button(
                                "🔬 Analyze",
                                id="analyze-button",
                                color="success",
                                className="w-100 mt-4"
                            )
                        ], md=2)
                    ])
                ])
            ], className="shadow-sm mb-4"),

            dcc.Loading(
                id="loading-analysis",
                type="default",
                children=html.Div(id="analysis-results")
            )
        ], style={"display": "none"})
    ], fluid=True, className="mb-5")


# Main layout
app.layout = dbc.Container([
    create_header(),
    create_input_section(),
    create_threshold_analysis_section(),
    create_detailed_analysis_section(),

    # Hidden stores for data
    dcc.Store(id='stock-data-store'),
    dcc.Store(id='corr-matrix-store'),
    dcc.Store(id='threshold-plot-store'),
    dcc.Store(id='threshold-stats-store'),
    dcc.Store(id='analysis-plots-store'),
    dcc.Store(id='analysis-data-store'),
], fluid=True, className="py-4")


# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    [Output('stock-data-store', 'data'),
     Output('corr-matrix-store', 'data'),
     Output('threshold-results', 'children'),
     Output('threshold-section', 'style'),
     Output('threshold-plot-store', 'data'),
     Output('threshold-stats-store', 'data'),
     Output('download-threshold-btn', 'disabled'),
     Output('detailed-section', 'style')],
    [Input('fetch-button', 'n_clicks')],
    [State('index-url', 'value'),
     State('force-fetch', 'value'),
     State('start-date', 'value'),
     State('end-date', 'value'),
     State('threshold-range', 'value')],
    prevent_initial_call=True
)
def fetch_data_and_analyze(n_clicks, index_url, force_fetch, start_date, end_date, threshold_range):
    """Fetch stock data and perform threshold analysis."""

    try:
        # Step 1: Fetch index
        index_df = nn.data.fetch_index(
            url=index_url,
            force_refresh=force_fetch,
            ticker_column="Symbol"
        )

        # Step 2: Fetch stock prices
        stock_prices = nn.data.fetch_stock_data(
            start_date=start_date,
            end_date=end_date,
            column="Close"
        )

        # Filter to date range if needed
        if start_date in stock_prices.index:
            stock_prices = stock_prices[start_date:]

        # Step 3: Remove columns with consecutive NaN values
        consec_nan = (stock_prices.isna() & stock_prices.isna().shift(1)).any(axis=0)
        stock_prices = stock_prices.loc[:, ~consec_nan]

        # Step 4: Calculate correlation matrix
        corr_mat = nn.graph.return_correlation_matrix(price_data=stock_prices)

        # Step 5: Build graphs for threshold range
        thresholds = np.arange(threshold_range[0], threshold_range[1], DEFAULT_THRESHOLD_STEP)

        # Create power-law analysis plots
        num_graphs = len(thresholds)
        num_cols = 2
        num_rows = (num_graphs + num_cols - 1) // num_cols

        fig_grid = make_subplots(
            rows=num_rows,
            cols=num_cols,
            subplot_titles=[f"Threshold={t:.2f}" for t in thresholds],
            specs=[[{"secondary_y": False} for _ in range(num_cols)] for _ in range(num_rows)]
        )

        # Update axes to log scale
        for i in range(1, num_graphs + 1):
            fig_grid.update_xaxes(type="log", row=(i - 1) // num_cols + 1, col=(i - 1) % num_cols + 1)
            fig_grid.update_yaxes(type="log", row=(i - 1) // num_cols + 1, col=(i - 1) % num_cols + 1)

        # Stats table data
        stats_data = []

        # Generate plots for each threshold
        for i, threshold in enumerate(thresholds):
            row = i // num_cols + 1
            col = i % num_cols + 1

            # Build graph
            G = nn.graph.build_correlation_graph(corr_mat, threshold=threshold)

            # Get degree distribution
            degrees, counts = nn.graph.degree_distribution(G)

            # Fit power law
            model, r_squared = nn.metrics.log_linear_fitting(degrees, counts)

            # Store stats
            N = G.number_of_nodes()
            L = G.number_of_edges()
            K = sum(dict(G.degree()).values()) / N if N > 0 else 0

            stats_data.append({
                'Threshold': f"{threshold:.2f}",
                'Nodes': N,
                'Edges': L,
                'Avg Degree': f"{K:.3f}",
                'R²': f"{r_squared:.4f}" if not np.isnan(r_squared) else "N/A"
            })

            # Skip plotting if no edges or insufficient data
            if len(degrees) == 0 or len(counts) == 0 or np.all(degrees == 0):
                continue

            # Add data points
            fig_grid.add_trace(
                go.Scatter(x=degrees, y=counts, mode='markers', name='Data', showlegend=False),
                row=row, col=col
            )

            # Add fitted line
            if not np.isnan(r_squared):
                x_fit = np.linspace(degrees[degrees > 0].min(), degrees.max(), 3, endpoint=True)
                y_fit = np.exp(model(np.log(x_fit)))
                fig_grid.add_trace(
                    go.Scatter(x=x_fit, y=y_fit, mode='lines', name='Power Law', showlegend=False),
                    row=row, col=col
                )

        fig_grid.update_layout(height=PLOT_HEIGHT_MULTIPLIER * num_rows, showlegend=False, title_text="Power Law Fitting Analysis")

        # Create stats table
        stats_df = pd.DataFrame(stats_data)
        stats_table = dbc.Table.from_dataframe(
            stats_df,
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
            className="mt-3"
        )

        results = html.Div([
            html.H5(f"✅ Successfully loaded {len(stock_prices.columns)} stocks", className="text-success mb-3"),
            html.H6("Network Statistics by Threshold:", className="mb-2"),
            stats_table,
            html.H6("Degree Distribution Plots (Log-Log Scale):", className="mt-4 mb-2"),
            dcc.Graph(figure=fig_grid)
        ])

        # Store data for later use
        stock_data_json = stock_prices.to_json(date_format='iso', orient='split')
        corr_matrix_json = corr_mat.to_json(orient='split')

        # Store plot and stats for download
        plot_json = fig_grid.to_json()
        stats_json = stats_df.to_json(orient='records')

        return stock_data_json, corr_matrix_json, results, {"display": "block"}, plot_json, stats_json, False, {"display": "block"}

    except Exception as e:
        error_alert = dbc.Alert(
            [
                html.H5("❌ Error", className="alert-heading"),
                html.P(f"An error occurred: {str(e)}")
            ],
            color="danger"
        )
        return None, None, error_alert, {"display": "block"}, None, None, True, {"display": "none"}


@app.callback(
    [Output('analysis-results', 'children'),
     Output('analysis-plots-store', 'data'),
     Output('analysis-data-store', 'data'),
     Output('download-analysis-btn', 'disabled')],
    [Input('analyze-button', 'n_clicks')],
    [State('corr-matrix-store', 'data'),
     State('analysis-threshold', 'value')],
    prevent_initial_call=True
)
def perform_detailed_analysis(n_clicks, corr_matrix_json, threshold):
    """Perform detailed network analysis for selected threshold."""

    if corr_matrix_json is None:
        return dbc.Alert("Please fetch data first!", color="warning"), None, None, True

    try:
        # Load correlation matrix
        corr_mat = pd.read_json(StringIO(corr_matrix_json), orient='split')

        # Build graph with selected threshold
        G = nn.graph.build_correlation_graph(corr_mat, threshold=threshold)

        # Get basic stats
        N = G.number_of_nodes()
        L = G.number_of_edges()
        K = sum(dict(G.degree()).values()) / N if N > 0 else 0

        # Compute centralities
        centralities = nn.metrics.compute_all_centralities(G)

        # Create centrality charts
        fig_degree = nn.visuals.create_centrality_chart(
            G,
            pd.Series(centralities["degree"]),
            title="Degree Centrality Analysis",
            top_n=TOP_N_CENTRALITY
        )

        fig_betweenness = nn.visuals.create_centrality_chart(
            G,
            pd.Series(centralities["betweenness"]),
            title="Betweenness Centrality Analysis",
            top_n=TOP_N_CENTRALITY
        )

        fig_closeness = nn.visuals.create_centrality_chart(
            G,
            pd.Series(centralities["closeness"]),
            title="Closeness Centrality Analysis",
            top_n=TOP_N_CENTRALITY
        )

        fig_eigenvector = nn.visuals.create_centrality_chart(
            G,
            pd.Series(centralities["eigenvector"]),
            title="Eigenvector Centrality Analysis",
            top_n=TOP_N_CENTRALITY
        )

        fig_pagerank = nn.visuals.create_centrality_chart(
            G,
            pd.Series(centralities["pagerank"]),
            title="PageRank Analysis",
            top_n=TOP_N_CENTRALITY
        )

        # Get top and least correlated nodes
        most_corr = nn.metrics.get_top_correlated_nodes(corr_mat, top_n=TOP_N_CORRELATION)
        least_corr = nn.metrics.get_top_correlated_nodes(corr_mat, top_n=TOP_N_CORRELATION, least=True)

        # Community detection
        communities = nn.metrics.compute_community_structure(G, algorithm="greedy")
        community_map = {}
        for node, comm_id in communities.items():
            if comm_id not in community_map:
                community_map[comm_id] = []
            community_map[comm_id].append(node)

        # Sort communities by size
        sorted_communities = sorted(community_map.items(), key=lambda x: len(x[1]), reverse=True)
        community_cards = []
        for comm_id, nodes in sorted_communities[:TOP_N_COMMUNITIES]:  # Show top communities
            if len(nodes) < 2:
                continue
            community_cards.append(
                dbc.Card([
                    dbc.CardHeader(f"Community {comm_id} ({len(nodes)} nodes)"),
                    dbc.CardBody([
                        html.P(", ".join(nodes[:10]) + ("..." if len(nodes) > 10 else ""))
                    ])
                ], className="mb-2")
            )

        # Clustering coefficient
        clustering_coeff = nn.metrics.compute_clustering_coefficient(G)
        clustering_values = list(clustering_coeff.values())
        global_clustering = nx.average_clustering(G)

        # Create results layout
        results = html.Div([
            # Network Statistics
            dbc.Card([
                dbc.CardHeader(html.H5("📊 Network Statistics")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6("Number of Nodes:"),
                            html.H4(f"{N}", className="text-info")
                        ], md=3),
                        dbc.Col([
                            html.H6("Number of Edges:"),
                            html.H4(f"{L}", className="text-info")
                        ], md=3),
                        dbc.Col([
                            html.H6("Average Degree:"),
                            html.H4(f"{K:.3f}", className="text-info")
                        ], md=3),
                        dbc.Col([
                            html.H6("Global Clustering:"),
                            html.H4(f"{global_clustering:.4f}", className="text-info")
                        ], md=3),
                    ])
                ])
            ], className="mb-4 shadow-sm"),

            # Top Correlated Nodes
            dbc.Card([
                dbc.CardHeader(html.H5("🔗 Correlation Analysis")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6("Most Correlated Pairs:"),
                            dbc.Table.from_dataframe(
                                most_corr.reset_index().rename(columns={'index': 'Pair', 0: 'Correlation'}),
                                striped=True,
                                bordered=True,
                                hover=True,
                                size="sm"
                            )
                        ], md=6),
                        dbc.Col([
                            html.H6("Least Correlated Pairs:"),
                            dbc.Table.from_dataframe(
                                least_corr.reset_index().rename(columns={'index': 'Pair', 0: 'Correlation'}),
                                striped=True,
                                bordered=True,
                                hover=True,
                                size="sm"
                            )
                        ], md=6),
                    ])
                ])
            ], className="mb-4 shadow-sm"),

            # Centrality Analyses
            html.H5("🎯 Centrality Measures", className="mt-4 mb-3"),
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_degree)
                ])
            ], className="mb-3 shadow-sm"),

            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_betweenness)
                ])
            ], className="mb-3 shadow-sm"),

            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_closeness)
                ])
            ], className="mb-3 shadow-sm"),

            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_eigenvector)
                ])
            ], className="mb-3 shadow-sm"),

            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_pagerank)
                ])
            ], className="mb-3 shadow-sm"),

            # Community Detection
            html.H5("👥 Community Structure", className="mt-4 mb-3"),
            dbc.Card([
                dbc.CardHeader(f"Detected {len(community_map)} communities"),
                dbc.CardBody(community_cards if community_cards else "No communities detected")
            ], className="mb-4 shadow-sm"),

            # Clustering Coefficient
            html.H5("🔄 Clustering Analysis", className="mt-4 mb-3"),
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.P(f"Mean Clustering Coefficient: {np.mean(clustering_values):.4f}"),
                            html.P(f"Median: {np.median(clustering_values):.4f}"),
                            html.P(f"Std Dev: {np.std(clustering_values):.4f}"),
                        ], md=6),
                        dbc.Col([
                            html.P(f"Min: {np.min(clustering_values):.4f}"),
                            html.P(f"Max: {np.max(clustering_values):.4f}"),
                            html.P(f"Global Clustering: {global_clustering:.4f}"),
                        ], md=6),
                    ])
                ])
            ], className="mb-4 shadow-sm"),
        ])

        # Store plots and data for download
        plots_data = {
            'degree': json.loads(fig_degree.to_json()),
            'betweenness': json.loads(fig_betweenness.to_json()),
            'closeness': json.loads(fig_closeness.to_json()),
            'eigenvector': json.loads(fig_eigenvector.to_json()),
            'pagerank': json.loads(fig_pagerank.to_json())
        }

        analysis_data = {
            'threshold': threshold,
            'network_stats': {
                'nodes': N,
                'edges': L,
                'avg_degree': float(K),
                'global_clustering': float(global_clustering)
            },
            'most_correlated': {str(k): v for k, v in most_corr.to_dict().items()},
            'least_correlated': {str(k): v for k, v in least_corr.to_dict().items()},
            'centralities': {
                'degree': centralities['degree'].nlargest(TOP_N_CENTRALITY).to_dict(),
                'betweenness': centralities['betweenness'].nlargest(TOP_N_CENTRALITY).to_dict(),
                'closeness': centralities['closeness'].nlargest(TOP_N_CENTRALITY).to_dict(),
                'eigenvector': centralities['eigenvector'].nlargest(TOP_N_CENTRALITY).to_dict(),
                'pagerank': centralities['pagerank'].nlargest(TOP_N_CENTRALITY).to_dict()
            },
            'communities': {int(k): v for k, v in community_map.items()},
            'clustering': {
                'mean': float(np.mean(clustering_values)),
                'median': float(np.median(clustering_values)),
                'std': float(np.std(clustering_values)),
                'min': float(np.min(clustering_values)),
                'max': float(np.max(clustering_values)),
                'global': float(global_clustering)
            }
        }

        return results, json.dumps(plots_data), json.dumps(analysis_data), False

    except Exception as e:
        error_alert = dbc.Alert(
            [
                html.H5("❌ Error", className="alert-heading"),
                html.P(f"An error occurred during analysis: {str(e)}")
            ],
            color="danger"
        )
        return error_alert, None, None, True


# ============================================================================
# DOWNLOAD CALLBACKS
# ============================================================================

@app.callback(
    Output('download-threshold', 'data'),
    Input('download-threshold-btn', 'n_clicks'),
    [State('threshold-plot-store', 'data'),
     State('threshold-stats-store', 'data')],
    prevent_initial_call=True
)
def download_threshold_analysis(n_clicks, plot_json, stats_json):
    """Download threshold analysis results as ZIP."""
    if plot_json is None or stats_json is None:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create ZIP file in memory
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add plot as PNG
        fig = go.Figure(json.loads(plot_json))
        img_bytes = fig.to_image(format="png", width=1920, height=1080)
        zip_file.writestr(f"threshold_analysis_{timestamp}.png", img_bytes)

        # Add stats as text
        stats_df = pd.read_json(StringIO(stats_json), orient='records')
        stats_text = f"""Nifty Network Analysis - Threshold Analysis
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'=' * 80}

NETWORK STATISTICS BY THRESHOLD
{'=' * 80}

{stats_df.to_string(index=False)}

{'=' * 80}
"""
        zip_file.writestr(f"threshold_stats_{timestamp}.txt", stats_text)

    # Encode ZIP for download
    zip_buffer.seek(0)
    zip_b64 = base64.b64encode(zip_buffer.read()).decode()

    return dict(content=zip_b64, filename=f"threshold_analysis_{timestamp}.zip", base64=True)


@app.callback(
    Output('download-analysis', 'data'),
    Input('download-analysis-btn', 'n_clicks'),
    [State('analysis-plots-store', 'data'),
     State('analysis-data-store', 'data')],
    prevent_initial_call=True
)
def download_detailed_analysis(n_clicks, plots_json, data_json):
    """Download detailed analysis results as ZIP with all plots and report."""
    if plots_json is None or data_json is None:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_data = json.loads(plots_json)
    analysis_data = json.loads(data_json)

    # Create ZIP file in memory
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add all centrality plots
        plot_names = ['degree', 'betweenness', 'closeness', 'eigenvector', 'pagerank']
        for plot_name in plot_names:
            fig = go.Figure(json.loads(json.dumps(plots_data[plot_name])))
            img_bytes = fig.to_image(format="png", width=1920, height=1080)
            zip_file.writestr(f"{plot_name}_centrality_{timestamp}.png", img_bytes)

        # Generate and add text report
        threshold = analysis_data['threshold']
        stats = analysis_data['network_stats']
        clustering = analysis_data['clustering']

        # Format centralities
        centrality_text = ""
        for cent_type, cent_data in analysis_data['centralities'].items():
            centrality_text += f"\n{cent_type.upper()} CENTRALITY (Top {TOP_N_CENTRALITY}):\n"
            centrality_text += "-" * 60 + "\n"
            for i, (node, value) in enumerate(cent_data.items(), 1):
                centrality_text += f"{i:2d}. {node:20s}: {value:.6f}\n"
            centrality_text += "\n"

        # Format communities
        community_text = "\nCOMMUNITY STRUCTURE:\n"
        community_text += "-" * 60 + "\n"
        for comm_id, nodes in sorted(analysis_data['communities'].items(),
                                     key=lambda x: len(x[1]), reverse=True)[:TOP_N_COMMUNITIES]:
            if len(nodes) >= 2:
                community_text += f"Community {comm_id} ({len(nodes)} nodes):\n"
                community_text += f"  {', '.join(nodes[:10])}"
                if len(nodes) > 10:
                    community_text += f" ... and {len(nodes) - 10} more"
                community_text += "\n\n"

        # Format correlations
        most_corr_text = "\nMOST CORRELATED PAIRS:\n"
        most_corr_text += "-" * 60 + "\n"
        for i, (pair, value) in enumerate(analysis_data['most_correlated'].items(), 1):
            most_corr_text += f"{i:2d}. {pair}: {value:.6f}\n"

        least_corr_text = "\nLEAST CORRELATED PAIRS:\n"
        least_corr_text += "-" * 60 + "\n"
        for i, (pair, value) in enumerate(analysis_data['least_correlated'].items(), 1):
            least_corr_text += f"{i:2d}. {pair}: {value:.6f}\n"

        report = f"""Nifty Network Analysis - Detailed Analysis Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'=' * 80}

NETWORK CONFIGURATION
{'=' * 80}
Correlation Threshold: {threshold:.2f}

NETWORK STATISTICS
{'=' * 80}
Number of Nodes:                {stats['nodes']}
Number of Edges:                {stats['edges']}
Average Degree:                 {stats['avg_degree']:.3f}
Global Clustering Coefficient:  {stats['global_clustering']:.4f}

CLUSTERING COEFFICIENTS
{'=' * 80}
Mean:                           {clustering['mean']:.4f}
Median:                         {clustering['median']:.4f}
Standard Deviation:             {clustering['std']:.4f}
Minimum:                        {clustering['min']:.4f}
Maximum:                        {clustering['max']:.4f}

{'=' * 80}
CENTRALITY ANALYSIS
{'=' * 80}
{centrality_text}
{'=' * 80}
CORRELATION ANALYSIS
{'=' * 80}
{most_corr_text}
{least_corr_text}
{'=' * 80}
{community_text}
{'=' * 80}

END OF REPORT
"""
        zip_file.writestr(f"network_analysis_{timestamp}.txt", report)

    # Encode ZIP for download
    zip_buffer.seek(0)
    zip_b64 = base64.b64encode(zip_buffer.read()).decode()

    return dict(content=zip_b64, filename=f"network_analysis_{timestamp}.zip", base64=True)


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    app.run(debug=DEBUG_MODE, host=SERVER_HOST, port=SERVER_PORT)
