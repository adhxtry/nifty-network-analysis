Community Detection Example
============================

This example demonstrates how to detect and analyze communities (clusters) in stock networks.

Complete Example
----------------

.. code-block:: python

   from niftynet import data, graph, metrics, visuals
   from datetime import datetime, timedelta
   import pandas as pd
   import networkx as nx

   # Fetch diverse set of stocks from different sectors
   tickers = [
       # Banking
       'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'AXISBANK.NS',
       # IT
       'TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS',
       # Energy
       'RELIANCE.NS', 'ONGC.NS', 'BPCL.NS', 'IOC.NS',
       # FMCG
       'HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS',
       # Auto
       'MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS',
       # Telecom
       'BHARTIARTL.NS', 'IDEA.NS'
   ]

   end_date = datetime.now()
   start_date = end_date - timedelta(days=365)

   print("Fetching stock data for community detection...")
   prices = data.fetch_stock_data(
       tickers,
       start_date.strftime('%Y-%m-%d'),
       end_date.strftime('%Y-%m-%d')
   )

   # Build correlation graph
   print("\nBuilding correlation graph...")
   threshold = 0.65  # Lower threshold to see more connections
   G, corr_matrix = graph.build_graph_from_prices(prices, threshold=threshold)

   print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
   print(f"Graph density: {nx.density(G):.4f}")

   # Detect communities
   print("\n" + "="*60)
   print("COMMUNITY DETECTION")
   print("="*60)

   communities = metrics.compute_community_structure(G)

   # Count communities
   unique_communities = set(communities.values())
   num_communities = len(unique_communities)

   print(f"\nFound {num_communities} communities")

   # Organize stocks by community
   community_groups = {}
   for node, comm_id in communities.items():
       if comm_id not in community_groups:
           community_groups[comm_id] = []
       community_groups[comm_id].append(node)

   # Display communities
   print("\nCommunity Composition:")
   print("-" * 60)

   for comm_id in sorted(community_groups.keys()):
       stocks = community_groups[comm_id]
       print(f"\nCommunity {comm_id} ({len(stocks)} stocks):")
       for stock in sorted(stocks):
           print(f"  - {stock}")

   # Analyze community statistics
   print("\n" + "="*60)
   print("COMMUNITY STATISTICS")
   print("="*60)

   for comm_id, stocks in community_groups.items():
       # Create subgraph for this community
       subgraph = G.subgraph(stocks)

       # Calculate statistics
       internal_edges = subgraph.number_of_edges()
       possible_edges = len(stocks) * (len(stocks) - 1) / 2
       internal_density = internal_edges / possible_edges if possible_edges > 0 else 0

       # Calculate external connections
       external_edges = 0
       for stock in stocks:
           for neighbor in G.neighbors(stock):
               if neighbor not in stocks:
                   external_edges += 1

       print(f"\nCommunity {comm_id}:")
       print(f"  Size: {len(stocks)} stocks")
       print(f"  Internal edges: {internal_edges}")
       print(f"  Internal density: {internal_density:.4f}")
       print(f"  External connections: {external_edges}")
       print(f"  Modularity contribution: High" if internal_density > 0.5 else "  Modularity contribution: Moderate")

   # Calculate overall modularity
   try:
       partition = {node: comm for node, comm in communities.items()}
       modularity = nx.community.modularity(G, community_groups.values())
       print(f"\nOverall Network Modularity: {modularity:.4f}")
       print("(Higher values indicate better community structure)")
   except:
       print("\nModularity calculation requires connected graph")

   # Visualize communities
   print("\n" + "="*60)
   print("CREATING VISUALIZATIONS")
   print("="*60)

   # Create network plot colored by community
   print("\nCreating community network plot...")
   node_colors = list(communities.values())

   fig_communities = visuals.create_network_plot(
       G,
       title=f"Stock Network Communities (threshold={threshold})",
       node_colors=node_colors
   )
   fig_communities.write_html("output_communities.html")
   print("Saved: output_communities.html")

   # Create correlation heatmap
   print("\nCreating correlation heatmap...")

   # Reorder matrix by communities
   ordered_stocks = []
   for comm_id in sorted(community_groups.keys()):
       ordered_stocks.extend(sorted(community_groups[comm_id]))

   # Reorder correlation matrix
   ordered_corr = corr_matrix.loc[ordered_stocks, ordered_stocks]

   fig_heatmap = visuals.create_correlation_heatmap(
       ordered_corr,
       title="Correlation Heatmap (ordered by communities)"
   )
   fig_heatmap.write_html("output_communities_heatmap.html")
   print("Saved: output_communities_heatmap.html")

   # Inter-community connections
   print("\n" + "="*60)
   print("INTER-COMMUNITY CONNECTIONS")
   print("="*60)

   inter_community_edges = {}

   for u, v in G.edges():
       comm_u = communities[u]
       comm_v = communities[v]

       if comm_u != comm_v:
           pair = tuple(sorted([comm_u, comm_v]))
           if pair not in inter_community_edges:
               inter_community_edges[pair] = []
           inter_community_edges[pair].append((u, v))

   print("\nConnections between communities:")
   for (comm1, comm2), edges in sorted(inter_community_edges.items()):
       print(f"\nCommunity {comm1} <-> Community {comm2}: {len(edges)} connections")
       for u, v in edges[:3]:  # Show first 3
           weight = G[u][v]['weight']
           print(f"  {u} -- {v} (correlation: {weight:.3f})")
       if len(edges) > 3:
           print(f"  ... and {len(edges) - 3} more")

   print("\nCommunity detection complete!")

Understanding Communities
--------------------------

What are Communities?
~~~~~~~~~~~~~~~~~~~~~

Communities (or clusters) are groups of stocks that are:

- More strongly correlated with each other than with stocks outside the group
- Often from the same sector or industry
- Subject to similar market forces or economic factors

Why Detect Communities?
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Sector Identification**: Automatically identify sector groups
2. **Diversification**: Build portfolios with stocks from different communities
3. **Risk Management**: Understand correlation clusters
4. **Market Structure**: Reveal hidden market relationships

Interpreting Results
---------------------

High Internal Density
~~~~~~~~~~~~~~~~~~~~~

Communities with high internal density (> 0.6):

- Strong intra-group correlations
- Likely from the same sector
- High co-movement risk

**Example**: All banking stocks in one community

Low External Connections
~~~~~~~~~~~~~~~~~~~~~~~~

Communities with few external connections:

- Isolated from other market segments
- Independent price movements
- Good for diversification

**Example**: Specialty pharmaceutical stocks

Bridge Stocks
~~~~~~~~~~~~~

Stocks connecting multiple communities:

- High betweenness centrality
- Cross-sector influence
- Conglomerates or diversified companies

**Example**: Reliance (operates in energy, retail, telecom)

Practical Applications
----------------------

Portfolio Construction
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Select one stock from each community for diversification
   diversified_portfolio = []

   for comm_id, stocks in community_groups.items():
       # Get highest market cap or most liquid stock
       selected = stocks[0]  # Simplified selection
       diversified_portfolio.append(selected)

   print("Diversified portfolio (one from each community):")
   for stock in diversified_portfolio:
       print(f"  - {stock}")

Sector Analysis
~~~~~~~~~~~~~~~

.. code-block:: python

   # Analyze largest community (likely a major sector)
   largest_comm_id = max(community_groups.keys(),
                         key=lambda x: len(community_groups[x]))
   largest_comm = community_groups[largest_comm_id]

   print(f"Largest community ({len(largest_comm)} stocks):")
   print("Likely represents a major sector")

   # Analyze their correlations
   comm_subgraph = G.subgraph(largest_comm)
   avg_degree = sum(dict(comm_subgraph.degree()).values()) / len(largest_comm)
   print(f"Average connections within group: {avg_degree:.1f}")

Risk Clustering
~~~~~~~~~~~~~~~

.. code-block:: python

   # Communities with high internal density = concentration risk
   high_risk_communities = []

   for comm_id, stocks in community_groups.items():
       subgraph = G.subgraph(stocks)
       density = nx.density(subgraph)

       if density > 0.7:
           high_risk_communities.append((comm_id, stocks, density))

   print("High concentration risk communities:")
   for comm_id, stocks, density in high_risk_communities:
       print(f"  Community {comm_id}: {len(stocks)} stocks, density={density:.3f}")

Advanced Techniques
-------------------

Hierarchical Communities
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Detect communities at different resolution levels
   from networkx.algorithms import community

   # Fine-grained communities
   communities_fine = metrics.compute_community_structure(G)

   # Coarse communities (merge related groups)
   # This requires implementing resolution parameter in Louvain

Temporal Community Evolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Analyze how communities change over time
   quarters = pd.date_range(start_date, end_date, freq='Q')

   for i in range(len(quarters)-1):
       quarter_prices = prices.loc[quarters[i]:quarters[i+1]]
       G_quarter, _ = graph.build_graph_from_prices(quarter_prices, threshold=0.65)
       communities_quarter = metrics.compute_community_structure(G_quarter)

       num_comms = len(set(communities_quarter.values()))
       print(f"Q{i+1}: {num_comms} communities")

Next Steps
----------

- Experiment with different threshold values
- Compare communities across time periods
- Map communities to actual market sectors
- Use communities for portfolio optimization
