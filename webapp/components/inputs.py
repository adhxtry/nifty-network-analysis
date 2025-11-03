"""
Input components for the Dash web application.
"""

from dash import dcc, html
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta


def create_input_section():
    """
    Create the input section with URL input, date pickers, and analyze button.

    Returns:
        Dash Bootstrap Components layout
    """
    default_end = datetime.now()
    default_start = default_end - timedelta(days=365)

    return dbc.Card([
        dbc.CardBody([
            html.H4("Data Configuration", className="card-title mb-4"),

            # CSV URL Input
            dbc.Row([
                dbc.Col([
                    html.Label("CSV URL with Nifty500 Tickers", className="fw-bold"),
                    html.P(
                        "Enter a URL to a CSV file containing stock tickers. "
                        "Default: Nifty500 index list",
                        className="text-muted small"
                    ),
                    dbc.Input(
                        id="csv-url-input",
                        type="url",
                        placeholder="https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
                        value="https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
                        className="mb-3"
                    )
                ], width=12)
            ]),

            # Date Range Pickers
            dbc.Row([
                dbc.Col([
                    html.Label("Start Date", className="fw-bold"),
                    dcc.DatePickerSingle(
                        id="start-date-picker",
                        date=default_start,
                        display_format="YYYY-MM-DD",
                        className="mb-3"
                    )
                ], width=12, md=6),

                dbc.Col([
                    html.Label("End Date", className="fw-bold"),
                    dcc.DatePickerSingle(
                        id="end-date-picker",
                        date=default_end,
                        display_format="YYYY-MM-DD",
                        className="mb-3"
                    )
                ], width=12, md=6)
            ]),

            # Correlation Threshold Slider
            dbc.Row([
                dbc.Col([
                    html.Label("Correlation Threshold (%)", className="fw-bold"),
                    html.P(
                        "Minimum correlation value to create edges in the network",
                        className="text-muted small"
                    ),
                    dcc.Slider(
                        id="correlation-threshold",
                        min=0,
                        max=100,
                        value=50,
                        marks={i: f"{i}%" for i in range(0, 101, 20)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], width=12, className="mb-3")
            ]),

            # Analyze Button
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        "Analyze Network",
                        id="analyze-button",
                        color="primary",
                        size="lg",
                        className="w-100"
                    )
                ], width=12, md=4)
            ])
        ])
    ], className="mb-4")
