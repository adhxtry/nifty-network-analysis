Advanced Analysis
=================

This tutorial covers advanced analysis techniques using NiftyNet.

Filtering Graphs
----------------

Filter graphs by various criteria to focus on meaningful connections:

.. code-block:: python

   from niftynet import graph

   # Build initial graph
   G, corr_matrix = graph.build_graph_from_prices(prices, threshold=0.5)

   # Filter nodes with minimum degree (remove isolated or weakly connected nodes)
   G_filtered = graph.filter_graph(G, min_degree=2)

   print(f"Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
   print(f"Filtered graph: {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")

Community Detection
-------------------

Detect communities (clusters) within the network to identify groups of related stocks:

.. code-block:: python

   from niftynet import metrics, visuals
   import plotly.graph_objects as go

   # Detect communities using Louvain method
   communities = metrics.compute_community_structure(G)

   num_communities = len(set(communities.values()))
   print(f"Found {num_communities} communities")

   # Print stocks in each community
   for community_id in range(num_communities):
       stocks = [node for node, comm in communities.items() if comm == community_id]
       print(f"Community {community_id}: {', '.join(stocks)}")

   # Visualize with community colors
   fig = visuals.create_network_plot(
       G,
       title=f"Network with {num_communities} Communities",
       node_colors=list(communities.values())
   )
   fig.show()

Comparing Correlation Methods
------------------------------

Different correlation methods can reveal different relationships:

.. code-block:: python

   from niftynet import graph
   import networkx as nx

   methods = ['pearson', 'kendall', 'spearman']
   results = {}

   for method in methods:
       G, corr = graph.build_graph_from_prices(
           prices,
           threshold=0.7,
           method=method
       )

       results[method] = {
           'nodes': G.number_of_nodes(),
           'edges': G.number_of_edges(),
           'density': nx.density(G),
           'avg_clustering': nx.average_clustering(G) if G.number_of_nodes() > 0 else 0
       }

   # Print comparison
   for method, stats in results.items():
       print(f"\n{method.capitalize()}:")
       print(f"  Nodes: {stats['nodes']}")
       print(f"  Edges: {stats['edges']}")
       print(f"  Density: {stats['density']:.4f}")
       print(f"  Avg Clustering: {stats['avg_clustering']:.4f}")

Analyzing Multiple Centrality Metrics
--------------------------------------

Compare different centrality metrics to identify key stocks:

.. code-block:: python

   from niftynet import metrics
   import pandas as pd

   # Compute all centrality metrics
   centrality = metrics.compute_all_centralities(G)

   # Create DataFrame for comparison
   df = pd.DataFrame({
       'Degree': centrality['degree'],
       'Betweenness': centrality['betweenness'],
       'Closeness': centrality['closeness'],
       'Eigenvector': centrality['eigenvector'],
       'PageRank': centrality['pagerank']
   })

   # Normalize for better comparison
   df_normalized = (df - df.min()) / (df.max() - df.min())

   # Get top stocks by each metric
   print("\nTop 3 stocks by each centrality metric:")
   for metric in df.columns:
       top_stocks = df[metric].nlargest(3)
       print(f"\n{metric}:")
       for stock, value in top_stocks.items():
           print(f"  {stock}: {value:.4f}")

Time-Based Network Evolution
-----------------------------

Analyze how network structure changes over time:

.. code-block:: python

   from niftynet import data, graph
   from datetime import datetime, timedelta
   import pandas as pd

   # Fetch 2 years of data
   tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
   prices = data.fetch_stock_data(tickers, '2022-01-01', '2024-01-01')

   # Analyze quarterly networks
   quarters = pd.date_range('2022-01-01', '2024-01-01', freq='Q')

   quarterly_stats = []
   for i in range(len(quarters) - 1):
       quarter_start = quarters[i]
       quarter_end = quarters[i + 1]

       # Get quarter data
       quarter_prices = prices.loc[quarter_start:quarter_end]

       # Build graph
       G, _ = graph.build_graph_from_prices(quarter_prices, threshold=0.7)

       stats = {
           'Quarter': f"Q{(i % 4) + 1} {quarter_start.year}",
           'Edges': G.number_of_edges(),
           'Density': nx.density(G),
           'Avg_Degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
       }
       quarterly_stats.append(stats)

   # Display results
   df_quarters = pd.DataFrame(quarterly_stats)
   print("\nQuarterly Network Evolution:")
   print(df_quarters.to_string(index=False))

Sector-Based Analysis
----------------------

Analyze correlations within and between sectors:

.. code-block:: python

   from niftynet import data, graph, visuals

   # Define stocks by sector
   sectors = {
       'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS'],
       'IT': ['TCS.NS', 'INFY.NS', 'WIPRO.NS'],
       'Energy': ['RELIANCE.NS', 'ONGC.NS', 'BPCL.NS']
   }

   # Flatten to list of tickers
   all_tickers = [ticker for sector_tickers in sectors.values() for ticker in sector_tickers]

   # Fetch data
   prices = data.fetch_stock_data(all_tickers, '2023-01-01', '2024-01-01')

   # Build graph
   G, corr_matrix = graph.build_graph_from_prices(prices, threshold=0.6)

   # Assign sector colors
   sector_colors = {}
   color_map = {'Banking': 0, 'IT': 1, 'Energy': 2}

   for sector, tickers in sectors.items():
       for ticker in tickers:
           sector_colors[ticker] = color_map[sector]

   # Visualize with sector colors
   node_colors = [sector_colors.get(node, -1) for node in G.nodes()]

   fig = visuals.create_network_plot(
       G,
       title="Stock Network by Sector",
       node_colors=node_colors
   )
   fig.show()

Advanced Filtering Techniques
------------------------------

Apply multiple filters to create focused networks:

.. code-block:: python

   from niftynet import graph
   import networkx as nx

   # Build initial graph with lower threshold
   G, corr_matrix = graph.build_graph_from_prices(prices, threshold=0.5)

   # Apply multiple filters
   # 1. Remove low-degree nodes
   G = graph.filter_graph(G, min_degree=2)

   # 2. Keep only largest connected component
   if G.number_of_nodes() > 0:
       largest_cc = max(nx.connected_components(G), key=len)
       G = G.subgraph(largest_cc).copy()

   # 3. Remove edges below a certain weight
   edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < 0.6]
   G.remove_edges_from(edges_to_remove)

   # 4. Remove any isolated nodes created
   G.remove_nodes_from(list(nx.isolates(G)))

   print(f"Final filtered graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

Next Steps
----------

- Explore :doc:`web_app_guide` for interactive analysis
- See :doc:`../examples/correlation_analysis` for complete examples
- Check out :doc:`../examples/community_detection` for advanced community analysis
