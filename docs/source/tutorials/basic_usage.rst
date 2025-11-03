Basic Usage Tutorial
====================

This tutorial covers the basic usage of NiftyNet library for stock network analysis.

Step 1: Fetching Stock Data
----------------------------

The first step is to fetch historical stock data using the `data` module:

.. code-block:: python

   from niftynet import data
   from datetime import datetime, timedelta
   
   # Define tickers for Nifty companies (add .NS suffix for NSE)
   tickers = [
       'RELIANCE.NS',
       'TCS.NS',
       'INFY.NS',
       'HDFCBANK.NS',
       'ICICIBANK.NS'
   ]
   
   # Fetch data for last year
   end_date = datetime.now()
   start_date = end_date - timedelta(days=365)
   
   prices = data.fetch_stock_data(
       tickers,
       start_date.strftime('%Y-%m-%d'),
       end_date.strftime('%Y-%m-%d')
   )
   
   print(f\"Fetched data shape: {prices.shape}\")

Step 2: Building Correlation Graphs
------------------------------------

Next, build a network graph based on correlation between stock prices:

.. code-block:: python

   from niftynet import graph
   
   # Build graph with 0.7 correlation threshold
   G, corr_matrix = graph.build_graph_from_prices(
       prices,
       threshold=0.7,
       method='pearson'
   )
   
   print(f\"Number of nodes: {G.number_of_nodes()}\")
   print(f\"Number of edges: {G.number_of_edges()}\")
   
   # Get graph summary
   summary = graph.get_graph_summary(G)
   print(summary)

Step 3: Computing Metrics
--------------------------

Compute various centrality metrics to identify important nodes:

.. code-block:: python

   from niftynet import metrics
   
   # Compute all centrality metrics
   centrality_scores = metrics.compute_all_centralities(G)
   
   # Get top 5 nodes by degree centrality
   top_nodes = metrics.get_top_nodes(
       centrality_scores['degree'],
       n=5
   )
   
   print(\"Top 5 most connected stocks:\")
   for ticker, score in top_nodes:
       print(f\"  {ticker}: {score:.4f}\")

Step 4: Visualization
----------------------

Create interactive visualizations with Plotly:

.. code-block:: python

   from niftynet import visuals
   
   # Create network plot
   fig = visuals.create_network_plot(
       G,
       title=\"Stock Correlation Network\",
       node_colors=centrality_scores['degree']
   )
   fig.show()
   
   # Create correlation heatmap
   heatmap = visuals.create_correlation_heatmap(
       corr_matrix,
       title=\"Stock Correlation Matrix\"
   )
   heatmap.show()
   
   # Create centrality bar chart
   bar_chart = visuals.create_centrality_bar_chart(
       centrality_scores['degree'],
       title=\"Degree Centrality\"
   )
   bar_chart.show()

Step 5: Saving Data
--------------------

Save your data for later analysis:

.. code-block:: python

   # Save price data to CSV
   output_path = data.save_to_csv(
       prices,
       output_dir=\"data\",
       filename=\"nifty_stocks_2024.csv\"
   )
   
   print(f\"Data saved to: {output_path}\")
   
   # Load data back
   loaded_prices = data.load_from_csv(output_path)

Complete Example
----------------

Here's a complete example combining all steps:

.. code-block:: python

   from niftynet import data, graph, metrics, visuals
   from datetime import datetime, timedelta
   
   # 1. Fetch data
   tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS']
   end_date = datetime.now()
   start_date = end_date - timedelta(days=365)
   
   prices = data.fetch_stock_data(
       tickers,
       start_date.strftime('%Y-%m-%d'),
       end_date.strftime('%Y-%m-%d')
   )
   
   # 2. Build graph
   G, corr_matrix = graph.build_graph_from_prices(prices, threshold=0.7)
   
   # 3. Compute metrics
   centrality = metrics.compute_all_centralities(G)
   
   # 4. Visualize
   fig = visuals.create_network_plot(
       G,
       title=\"Stock Correlation Network\",
       node_colors=centrality['degree']
   )
   fig.show()
   
   # 5. Save data
   data.save_to_csv(prices, output_dir=\"data\", filename=\"stocks.csv\")
   
   print(\"Analysis complete!\")

Next Steps
----------

- Learn about :doc:dvanced_analysis techniques
- Explore the :doc:../examples/correlation_analysis example
- Try the :doc:web_app_guide for interactive analysis
