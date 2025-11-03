Correlation Analysis Example
============================

This example demonstrates complete correlation analysis workflow for Nifty stocks.

Complete Example Code
----------------------

.. code-block:: python

   from niftynet import data, graph, visuals, metrics
   from datetime import datetime, timedelta
   import pandas as pd

   # Configuration
   tickers = [
       'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS',
       'ICICIBANK.NS', 'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS',
       'BHARTIARTL.NS', 'KOTAKBANK.NS'
   ]

   end_date = datetime.now()
   start_date = end_date - timedelta(days=365)

   # Step 1: Fetch Data
   print("Fetching stock data...")
   prices = data.fetch_stock_data(
       tickers,
       start_date.strftime('%Y-%m-%d'),
       end_date.strftime('%Y-%m-%d')
   )

   print(f"Fetched data shape: {prices.shape}")
   print(f"Date range: {prices.index[0]} to {prices.index[-1]}")

   # Step 2: Save Data
   output_path = data.save_to_csv(
       prices,
       output_dir="data",
       filename="nifty_top10_analysis.csv"
   )
   print(f"Data saved to: {output_path}")

   # Step 3: Compute Correlation Matrix
   print("\nComputing correlation matrix...")
   corr_matrix = graph.compute_correlation_matrix(prices, method='pearson')

   # Display top correlations
   print("\nTop 5 Correlations:")
   corr_pairs = []
   for i in range(len(corr_matrix.columns)):
       for j in range(i+1, len(corr_matrix.columns)):
           corr_pairs.append({
               'Stock1': corr_matrix.columns[i],
               'Stock2': corr_matrix.columns[j],
               'Correlation': corr_matrix.iloc[i, j]
           })

   df_corr = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False)
   print(df_corr.head(5).to_string(index=False))

   # Step 4: Build Graph with Multiple Thresholds
   thresholds = [0.6, 0.7, 0.8]

   for threshold in thresholds:
       print(f"\n--- Threshold: {threshold} ---")
       G, _ = graph.build_graph_from_prices(prices, threshold=threshold)

       summary = graph.get_graph_summary(G)
       print(summary)

   # Step 5: Build Final Graph
   threshold = 0.7
   G, corr_matrix = graph.build_graph_from_prices(
       prices,
       threshold=threshold,
       method='pearson'
   )

   # Step 6: Create Visualizations
   print(f"\nCreating visualizations...")

   # 6a. Network Plot
   fig_network = visuals.create_network_plot(
       G,
       title=f"Stock Correlation Network (threshold={threshold})"
   )
   fig_network.write_html("output_network.html")
   print("Network plot saved to output_network.html")

   # 6b. Correlation Heatmap
   fig_heatmap = visuals.create_correlation_heatmap(
       corr_matrix,
       title="Stock Correlation Heatmap"
   )
   fig_heatmap.write_html("output_heatmap.html")
   print("Heatmap saved to output_heatmap.html")

   # Step 7: Analyze Network Structure
   print("\n--- Network Analysis ---")

   # Degree distribution
   degrees = dict(G.degree())
   print("\nDegree Centrality:")
   for node, degree in sorted(degrees.items(), key=lambda x: x[1], reverse=True):
       print(f"  {node}: {degree} connections")

   # Create degree distribution plot
   fig_degree = visuals.create_degree_distribution(
       G,
       title="Degree Distribution"
   )
   fig_degree.write_html("output_degree_dist.html")
   print("Degree distribution saved to output_degree_dist.html")

   # Step 8: Compare Correlation Methods
   print("\n--- Comparing Correlation Methods ---")

   methods = ['pearson', 'kendall', 'spearman']
   method_results = {}

   for method in methods:
       G_method, corr_method = graph.build_graph_from_prices(
           prices,
           threshold=0.7,
           method=method
       )

       method_results[method] = {
           'edges': G_method.number_of_edges(),
           'density': round(G_method.number_of_edges() /
                          (G_method.number_of_nodes() * (G_method.number_of_nodes() - 1) / 2), 4)
       }

       print(f"\n{method.capitalize()}:")
       print(f"  Edges: {method_results[method]['edges']}")
       print(f"  Density: {method_results[method]['density']}")

   print("\nAnalysis complete!")

Expected Output
---------------

Running this example will produce:

1. **Console Output**:
   - Data fetch progress
   - Top correlations
   - Network statistics for different thresholds
   - Degree centrality rankings

2. **HTML Files**:
   - ``output_network.html``: Interactive network visualization
   - ``output_heatmap.html``: Correlation heatmap
   - ``output_degree_dist.html``: Degree distribution chart

3. **Data Files**:
   - ``data/nifty_top10_analysis.csv``: Historical price data

Interpretation
--------------

High Correlations
~~~~~~~~~~~~~~~~~

Stocks with correlation > 0.8 typically:

- Belong to the same sector
- Are affected by similar market forces
- Have related business models

Example: Banks (HDFCBANK.NS, ICICIBANK.NS, SBIN.NS) often show high correlation.

Network Density
~~~~~~~~~~~~~~~

- **Low density** (< 0.3): Diverse portfolio with independent stocks
- **Medium density** (0.3-0.6): Some sector clustering
- **High density** (> 0.6): Highly correlated stocks, similar risk exposure

Degree Centrality
~~~~~~~~~~~~~~~~~

Stocks with high degree are:

- **Market leaders**: Often large-cap stocks
- **Sector representatives**: Core stocks in their sector
- **Risk indicators**: High connectivity suggests systemic risk

Variations
----------

Sector-Specific Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Focus on banking sector
   bank_tickers = [
       'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS',
       'KOTAKBANK.NS', 'AXISBANK.NS'
   ]

   # Same analysis workflow
   prices = data.fetch_stock_data(bank_tickers, start_date, end_date)
   G, corr = graph.build_graph_from_prices(prices, threshold=0.6)

Time-Window Analysis
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Analyze different time windows
   windows = [30, 90, 180, 365]  # days

   for days in windows:
       window_start = end_date - timedelta(days=days)
       window_prices = prices.loc[window_start:]

       G, _ = graph.build_graph_from_prices(window_prices, threshold=0.7)
       print(f"{days} days: {G.number_of_edges()} edges")
