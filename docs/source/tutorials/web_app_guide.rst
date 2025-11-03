Web Application Guide
=====================

This guide covers how to use the Dash-based web application for interactive stock network analysis.

Starting the Application
-------------------------

To start the web application:

.. code-block:: bash

   cd webapp
   uv run python app.py

The application will start on http://127.0.0.1:8050/

Open your web browser and navigate to this URL.

User Interface Overview
-----------------------

The web application consists of three main sections:

1. **Input Section** - Configure your analysis parameters
2. **Visualization Section** - View interactive network graphs
3. **Statistics Section** - See network metrics and summaries

Input Parameters
----------------

CSV URL Input
~~~~~~~~~~~~~

Enter the URL to a CSV file containing stock tickers. The default URL points to the Nifty 500 index:

.. code-block:: text

   https://archives.nseindia.com/content/indices/ind_nifty500list.csv

Your CSV should have a column named ``Symbol``, ``Ticker``, or similar containing stock ticker symbols.

Date Range
~~~~~~~~~~

Select the start and end dates for historical data:

- **Start Date**: Beginning of the analysis period
- **End Date**: End of the analysis period

.. note::
   Longer date ranges provide more data but take longer to process.

Correlation Threshold
~~~~~~~~~~~~~~~~~~~~~

Use the slider to set the correlation threshold (0-100%):

- **Lower values** (50-70%): More connections, denser network
- **Higher values** (70-90%): Fewer connections, only strong correlations

.. tip::
   Start with 70% and adjust based on results. If you see too many connections, increase the threshold.

Running an Analysis
-------------------

Step-by-Step Process
~~~~~~~~~~~~~~~~~~~~

1. **Configure Parameters**

   - Enter CSV URL (or use default)
   - Select date range (e.g., last year)
   - Set correlation threshold (e.g., 70%)

2. **Click "Analyze Network"**

   The app will:

   - Fetch stock data from Yahoo Finance
   - Calculate correlation matrix
   - Build network graph
   - Generate visualization

3. **View Results**

   - Interactive network plot
   - Graph statistics (nodes, edges, density)
   - Saved data file path

Example Analysis
~~~~~~~~~~~~~~~~

Let's analyze the top Nifty 50 stocks over the last 6 months with 75% correlation threshold:

1. **CSV URL**: Use default Nifty 500 URL (first 50 stocks will be used)
2. **Start Date**: 6 months ago
3. **End Date**: Today
4. **Threshold**: 75%
5. Click **Analyze Network**

Results will show stocks that are highly correlated (move together) over this period.

Understanding the Visualization
--------------------------------

Network Plot Features
~~~~~~~~~~~~~~~~~~~~~

The interactive network plot includes:

- **Nodes**: Each node represents a stock
- **Edges**: Lines connecting stocks with high correlation
- **Colors**: Node colors can indicate centrality or communities
- **Layout**: Force-directed layout positions related stocks closer

Interacting with the Plot
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Hover**: Mouse over nodes to see stock names
- **Zoom**: Scroll to zoom in/out
- **Pan**: Click and drag to pan around
- **Reset**: Double-click to reset view

Graph Statistics
~~~~~~~~~~~~~~~~

Below the visualization, you'll see:

- **Number of Nodes**: Total stocks in the network
- **Number of Edges**: Total connections between stocks
- **Network Density**: How connected the network is (0-1)
- **Data File**: Path to saved CSV data

Interpreting Results
---------------------

High Connectivity
~~~~~~~~~~~~~~~~~

Stocks with many connections (high degree) are:

- Central to the market
- Influenced by or influence many other stocks
- Often large-cap or sector leaders

Clusters
~~~~~~~~

Groups of tightly connected stocks often represent:

- Stocks in the same sector (e.g., all banks)
- Stocks affected by similar factors
- Companies with related business models

Isolated Nodes
~~~~~~~~~~~~~~

Stocks with few or no connections:

- Move independently of others
- May be in unique sectors
- Could have different risk profiles

Performance Tips
----------------

For Faster Analysis
~~~~~~~~~~~~~~~~~~~

1. **Limit Stocks**: App automatically limits to first 50 tickers
2. **Shorter Date Range**: Use 3-6 months instead of multiple years
3. **Higher Threshold**: 75-80% for fewer connections

For Better Insights
~~~~~~~~~~~~~~~~~~~

1. **Sector-Specific Analysis**: Use CSV with stocks from one sector
2. **Longer Time Period**: 1-2 years for stable correlations
3. **Lower Threshold**: 60-70% to see more relationships

Troubleshooting
---------------

No Connections Shown
~~~~~~~~~~~~~~~~~~~~

**Problem**: Graph shows "No connections found"

**Solutions**:

- Lower the correlation threshold (try 60% or 50%)
- Increase the date range
- Check that stocks have sufficient data

Slow Loading
~~~~~~~~~~~~

**Problem**: Analysis takes a long time

**Solutions**:

- Use fewer stocks (app limits to 50 automatically)
- Reduce date range
- Check internet connection (data is fetched from Yahoo Finance)

Error Messages
~~~~~~~~~~~~~~

**CSV URL Error**: Check that URL is accessible and contains valid ticker data

**Data Fetch Error**: Some tickers may not be available on Yahoo Finance

Advanced Usage
--------------

Custom CSV Files
~~~~~~~~~~~~~~~~

Create your own CSV file with specific stocks:

.. code-block:: text

   Symbol
   RELIANCE.NS
   TCS.NS
   INFY.NS
   HDFCBANK.NS

Upload to a public URL (GitHub, Google Drive, etc.) and use that URL in the app.

Analyzing Specific Sectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Find a sector-specific stock list
2. Create CSV with those tickers
3. Host publicly and use URL in app
4. Analyze with appropriate threshold (60-70% for sectors)

Comparing Time Periods
~~~~~~~~~~~~~~~~~~~~~~~

Run multiple analyses with different date ranges:

1. Analysis 1: Last 3 months
2. Analysis 2: Last 6 months
3. Analysis 3: Last 12 months
4. Compare how network structure changes

Exporting Data
--------------

Data Files Location
~~~~~~~~~~~~~~~~~~~

Analyzed data is automatically saved to:

.. code-block:: text

   data/stock_data_YYYY-MM-DD_to_YYYY-MM-DD.csv

These files contain the historical price data used for analysis.

Using Exported Data
~~~~~~~~~~~~~~~~~~~

You can load saved data in Python:

.. code-block:: python

   from niftynet import data

   # Load exported data
   prices = data.load_from_csv('data/stock_data_2024-01-01_to_2024-12-31.csv')

   # Continue with analysis
   from niftynet import graph
   G, corr = graph.build_graph_from_prices(prices, threshold=0.7)

Next Steps
----------

- Try analyzing different sectors
- Experiment with various thresholds
- Compare results across different time periods
- Use :doc:`basic_usage` for programmatic analysis
- See :doc:`advanced_analysis` for more complex techniques
