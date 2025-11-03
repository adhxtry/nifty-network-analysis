Centrality Metrics Example
===========================

This example demonstrates how to compute and analyze various centrality metrics.

Complete Example
----------------

.. code-block:: python

   from niftynet import data, graph, metrics, visuals
   from datetime import datetime, timedelta
   import pandas as pd

   # Fetch data for major Nifty stocks
   tickers = [
       'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS',
       'ICICIBANK.NS', 'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS',
       'BHARTIARTL.NS', 'KOTAKBANK.NS', 'LT.NS', 'AXISBANK.NS'
   ]

   end_date = datetime.now()
   start_date = end_date - timedelta(days=365)

   print("Fetching stock data...")
   prices = data.fetch_stock_data(
       tickers,
       start_date.strftime('%Y-%m-%d'),
       end_date.strftime('%Y-%m-%d')
   )

   # Build correlation graph
   print("\nBuilding correlation graph...")
   G, corr_matrix = graph.build_graph_from_prices(prices, threshold=0.7)

   print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

   # Compute all centrality metrics
   print("\n=== Computing Centrality Metrics ===\n")

   centrality_scores = metrics.compute_all_centralities(G)

   # Create DataFrame for analysis
   df_centrality = pd.DataFrame({
       'Degree': centrality_scores['degree'],
       'Betweenness': centrality_scores['betweenness'],
       'Closeness': centrality_scores['closeness'],
       'Eigenvector': centrality_scores['eigenvector'],
       'PageRank': centrality_scores['pagerank']
   })

   # Display all centrality scores
   print("Centrality Scores:")
   print(df_centrality.round(4).to_string())

   # Analyze each centrality metric
   print("\n" + "="*60)
   print("CENTRALITY ANALYSIS")
   print("="*60)

   # 1. Degree Centrality
   print("\n1. DEGREE CENTRALITY (Most Connected Stocks)")
   print("-" * 50)
   top_degree = metrics.get_top_nodes(centrality_scores['degree'], n=5)
   for i, (node, score) in enumerate(top_degree, 1):
       print(f"{i}. {node:15s} : {score:.4f} ({int(score * (G.number_of_nodes()-1))} connections)")

   # 2. Betweenness Centrality
   print("\n2. BETWEENNESS CENTRALITY (Bridge Stocks)")
   print("-" * 50)
   top_betweenness = metrics.get_top_nodes(centrality_scores['betweenness'], n=5)
   for i, (node, score) in enumerate(top_betweenness, 1):
       print(f"{i}. {node:15s} : {score:.4f}")

   # 3. Closeness Centrality
   print("\n3. CLOSENESS CENTRALITY (Central Position)")
   print("-" * 50)
   top_closeness = metrics.get_top_nodes(centrality_scores['closeness'], n=5)
   for i, (node, score) in enumerate(top_closeness, 1):
       print(f"{i}. {node:15s} : {score:.4f}")

   # 4. Eigenvector Centrality
   print("\n4. EIGENVECTOR CENTRALITY (Connected to Important Stocks)")
   print("-" * 50)
   top_eigenvector = metrics.get_top_nodes(centrality_scores['eigenvector'], n=5)
   for i, (node, score) in enumerate(top_eigenvector, 1):
       print(f"{i}. {node:15s} : {score:.4f}")

   # 5. PageRank
   print("\n5. PAGERANK (Overall Importance)")
   print("-" * 50)
   top_pagerank = metrics.get_top_nodes(centrality_scores['pagerank'], n=5)
   for i, (node, score) in enumerate(top_pagerank, 1):
       print(f"{i}. {node:15s} : {score:.4f}")

   # Clustering coefficient
   print("\n6. CLUSTERING COEFFICIENT (Local Clustering)")
   print("-" * 50)
   clustering = metrics.compute_clustering_coefficient(G)
   top_clustering = sorted(clustering.items(), key=lambda x: x[1], reverse=True)[:5]
   for i, (node, score) in enumerate(top_clustering, 1):
       print(f"{i}. {node:15s} : {score:.4f}")

   # Create visualizations
   print("\n" + "="*60)
   print("CREATING VISUALIZATIONS")
   print("="*60)

   # Visualize with different centrality metrics
   centrality_types = ['degree', 'betweenness', 'pagerank']

   for cent_type in centrality_types:
       print(f"\nCreating {cent_type} centrality plot...")

       # Network plot
       fig_network = visuals.create_network_plot(
           G,
           title=f"Network colored by {cent_type.capitalize()} Centrality",
           node_colors=centrality_scores[cent_type]
       )
       fig_network.write_html(f"output_{cent_type}_network.html")

       # Bar chart
       fig_bar = visuals.create_centrality_bar_chart(
           centrality_scores[cent_type],
           title=f"{cent_type.capitalize()} Centrality Rankings"
       )
       fig_bar.write_html(f"output_{cent_type}_bar.html")

       print(f"  - Saved {cent_type}_network.html")
       print(f"  - Saved {cent_type}_bar.html")

   # Correlation analysis
   print("\nAnalyzing centrality correlations...")
   centrality_corr = df_centrality.corr()

   print("\nCentrality Metric Correlations:")
   print(centrality_corr.round(3).to_string())

   print("\nAnalysis complete!")

Understanding Centrality Metrics
---------------------------------

Degree Centrality
~~~~~~~~~~~~~~~~~

**What it measures**: Number of direct connections

**Interpretation**:
- High degree = Stock is correlated with many others
- Indicates market influence and sector representation
- Often corresponds to large-cap stocks

**Use case**: Identify stocks that move with many others

Betweenness Centrality
~~~~~~~~~~~~~~~~~~~~~~~

**What it measures**: How often a node lies on shortest paths between other nodes

**Interpretation**:
- High betweenness = Stock acts as a "bridge"
- Connects different parts of the network
- May represent cross-sector influence

**Use case**: Find stocks that connect different market segments

Closeness Centrality
~~~~~~~~~~~~~~~~~~~~

**What it measures**: Average distance to all other nodes

**Interpretation**:
- High closeness = Stock is "central" to the network
- Can quickly influence or be influenced by others
- Represents overall market connectivity

**Use case**: Identify stocks with broad market influence

Eigenvector Centrality
~~~~~~~~~~~~~~~~~~~~~~~

**What it measures**: Quality of connections (connected to important nodes)

**Interpretation**:
- High eigenvector = Connected to other important stocks
- Represents influence in high-value networks
- Often highlights market leaders

**Use case**: Find stocks connected to market leaders

PageRank
~~~~~~~~

**What it measures**: Google's PageRank algorithm applied to stocks

**Interpretation**:
- Combines quantity and quality of connections
- Represents overall importance in network
- Accounts for both direct and indirect connections

**Use case**: Overall stock importance ranking

Clustering Coefficient
~~~~~~~~~~~~~~~~~~~~~~

**What it measures**: How clustered a node's neighbors are

**Interpretation**:
- High clustering = Stock's connections are also connected
- Indicates sector cohesion
- Low clustering = Stock bridges different groups

**Use case**: Identify sector boundaries and bridges

Practical Applications
----------------------

Portfolio Diversification
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Find stocks with low degree centrality (more independent)
   low_degree = {k: v for k, v in centrality_scores['degree'].items()
                 if v < 0.3}

   print("Low correlation stocks (good for diversification):")
   for stock in low_degree:
       print(f"  - {stock}")

Risk Assessment
~~~~~~~~~~~~~~~

.. code-block:: python

   # High degree = high systemic risk
   high_degree = {k: v for k, v in centrality_scores['degree'].items()
                  if v > 0.7}

   print("High systemic risk stocks:")
   for stock, score in sorted(high_degree.items(),
                               key=lambda x: x[1], reverse=True):
       print(f"  - {stock}: {score:.3f}")

Sector Leaders
~~~~~~~~~~~~~~

.. code-block:: python

   # High eigenvector = sector leaders
   leaders = metrics.get_top_nodes(centrality_scores['eigenvector'], n=3)

   print("Sector leaders (high eigenvector centrality):")
   for stock, score in leaders:
       print(f"  - {stock}: {score:.4f}")
