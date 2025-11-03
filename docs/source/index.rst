.. nifty-network-analysis documentation master file, created by
   sphinx-quickstart on Mon Nov  3 15:54:24 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Nifty Network Analysis Documentation!
=================================================

.. image:: https://img.shields.io/badge/python-3.12%2B-blue
   :target: https://www.python.org/downloads/
   :alt: Python 3.12+

.. image:: https://img.shields.io/badge/license-MIT-green
   :alt: MIT License

**NiftyNet** is a Python library for analyzing and visualizing network relationships
between Nifty companies using stock market data. It provides tools to build correlation-based
graphs, compute network metrics, and create interactive visualizations.

Features
--------

✨ **Core Functionality**

- 📊 Fetch historical stock data using yfinance
- 🔗 Build correlation-based network graphs (Pearson, Kendall, Spearman)
- 📈 Compute network centrality metrics (degree, betweenness, closeness, eigenvector, PageRank)
- 🎨 Create interactive visualizations with Plotly
- 🌐 Dash-based web application for easy analysis
- 💾 Export data to CSV format
- 🎯 Filter graphs by correlation threshold
- 👥 Detect community structure

Installation
------------

**Using uv (recommended):**

.. code-block:: bash

   git clone https://github.com/adhxtry/nifty-network-analysis.git
   cd nifty-network-analysis
   uv sync

**Using pip:**

.. code-block:: bash

   pip install nifty-network-analysis

Quick Start
-----------

**Basic Usage:**

.. code-block:: python

   from niftynet import data, graph, metrics, visuals

   # Fetch stock data for Nifty companies
   tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS']
   prices = data.fetch_stock_data(tickers, '2023-01-01', '2024-01-01')

   # Build correlation graph with 0.7 threshold
   G, corr_matrix = graph.build_graph_from_prices(prices, threshold=0.7)

   # Compute centrality metrics
   centrality = metrics.compute_all_centralities(G)

   # Create interactive visualization
   fig = visuals.create_network_plot(G, title="Stock Correlation Network")
   fig.show()

**Web Application:**

Run the Dash web application for an interactive experience:

.. code-block:: bash

   cd webapp
   uv run python app.py

Then open your browser to http://127.0.0.1:8050

Tutorials
---------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   tutorials/basic_usage
   tutorials/advanced_analysis
   tutorials/web_app_guide

API Documentation
-----------------

.. toctree::
   :maxdepth: 3
   :caption: API Reference:

   modules/data
   modules/graph
   modules/metrics
   modules/visuals

Examples
--------

.. toctree::
   :maxdepth: 2
   :caption: Examples:

   examples/correlation_analysis
   examples/centrality_metrics
   examples/community_detection

Contributing
------------

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: ``uv run pytest tests/``
5. Submit a pull request

Development Setup
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Clone repository
   git clone https://github.com/adhxtry/nifty-network-analysis.git
   cd nifty-network-analysis

   # Install development dependencies
   uv sync --extra dev

   # Run tests
   uv run pytest tests/ -v

   # Build documentation
   cd docs
   uv run sphinx-build -b html source build

License
-------

This project is licensed under the MIT License.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
