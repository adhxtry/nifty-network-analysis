# Nifty Network Analysis

A comprehensive Python project for analyzing and visualizing network relationships between Nifty companies using stock market data.

## Features

- 📊 **Stock Data Fetching**: Fetch historical stock data using yfinance
- 🕸️ **Network Analysis**: Build correlation-based graphs using NetworkX
- 📈 **Centrality Metrics**: Compute degree, betweenness, closeness, and eigenvector centrality
- 🎨 **Interactive Visualizations**: Create beautiful plots with Plotly
- 🌐 **Web Application**: User-friendly Dash web interface
- 📚 **Documentation**: Complete API documentation with Sphinx

## Project Structure

```
nifty-network-analysis/
├── niftynet/              # Core Python library
│   ├── data.py           # Data fetching and storage
│   ├── graph.py          # Graph construction
│   ├── metrics.py        # Network metrics computation
│   └── visuals.py        # Visualization functions
├── webapp/                # Dash web application
│   ├── app.py            # Main application
│   └── components/       # Reusable UI components
├── tests/                 # Unit tests
├── docs/                  # Sphinx documentation
└── pyproject.toml        # Project configuration
```

## Installation

### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/adhxtry/nifty-network-analysis.git
cd nifty-network-analysis

# Install dependencies using uv
uv sync
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/adhxtry/nifty-network-analysis.git
cd nifty-network-analysis

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### Install Development Dependencies

```bash
uv sync --extra dev
```

## Quick Start

### Using the Python Library

```python
from niftynet import data, graph, metrics, visuals

# 1. Fetch stock data
tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS']
prices = data.fetch_stock_data(tickers, '2023-01-01', '2024-01-01')

# 2. Save data to CSV
data.save_to_csv(prices, output_dir='data', filename='nifty_stocks.csv')

# 3. Build correlation graph
G, corr_matrix = graph.build_graph_from_prices(prices, threshold=0.5)

# 4. Compute centrality metrics
centralities = metrics.compute_all_centralities(G)
print(centralities)

# 5. Create visualization
fig = visuals.create_network_plot(G, title='Nifty Stock Network')
fig.show()
```

### Running the Web Application

```bash
# Start the Dash web app
python webapp/app.py
```

Then open your browser to: http://127.0.0.1:8050

#### Using the Web App

1. **Enter CSV URL**: Provide a URL to a CSV file containing Nifty500 tickers
   - Default: https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv

2. **Select Date Range**: Choose start and end dates for historical data

3. **Set Correlation Threshold**: Adjust the minimum correlation for network edges

4. **Click "Analyze Network"**: View the interactive network visualization

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=niftynet --cov-report=html
```

## Documentation

### Building Documentation Locally

```bash
cd docs
make html
```

The documentation will be available at `docs/build/html/index.html`

### Viewing Documentation Online

Documentation is automatically built and deployed to GitHub Pages on every push to master:

👉 [https://adhxtry.github.io/nifty-network-analysis/](https://adhxtry.github.io/nifty-network-analysis/)

### Setting Up GitHub Pages

To enable GitHub Pages for your fork:

1. Go to your repository on GitHub
2. Click **Settings** → **Pages**
3. Under **Source**, select **GitHub Actions**
4. The documentation will be automatically deployed on the next push

## API Reference

### Data Module (`niftynet.data`)

- `fetch_stock_data()` - Fetch historical stock prices
- `save_to_csv()` - Save DataFrame to CSV
- `load_from_csv()` - Load data from CSV
- `fetch_and_save()` - Fetch and save in one operation

### Graph Module (`niftynet.graph`)

- `compute_correlation_matrix()` - Calculate correlation matrix
- `build_correlation_graph()` - Create graph from correlations
- `filter_graph()` - Remove low-degree nodes
- `get_graph_summary()` - Get graph statistics

### Metrics Module (`niftynet.metrics`)

- `compute_degree_centrality()` - Node degree centrality
- `compute_betweenness_centrality()` - Betweenness centrality
- `compute_closeness_centrality()` - Closeness centrality
- `compute_eigenvector_centrality()` - Eigenvector centrality
- `compute_pagerank()` - PageRank scores
- `compute_all_centralities()` - All metrics in one DataFrame

### Visuals Module (`niftynet.visuals`)

- `create_network_plot()` - Interactive network visualization
- `create_correlation_heatmap()` - Correlation matrix heatmap
- `create_centrality_bar_chart()` - Bar chart for rankings
- `create_degree_distribution()` - Degree distribution histogram

## Examples

### Example 1: Analyze Top 10 Nifty Stocks

```python
from niftynet import data, graph, metrics

# Fetch data for top stocks
tickers = [f'{stock}.NS' for stock in ['RELIANCE', 'TCS', 'HDFCBANK',
                                         'INFY', 'ICICIBANK', 'HINDUNILVR',
                                         'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK']]
prices = data.fetch_stock_data(tickers, '2023-01-01', '2024-01-01')

# Build and analyze network
G, _ = graph.build_graph_from_prices(prices, threshold=0.6)
centralities = metrics.compute_all_centralities(G)

# Show most central stocks
print("Top 5 by Degree Centrality:")
print(centralities.nlargest(5, 'degree')[['degree']])
```

### Example 2: Custom Visualization

```python
from niftynet import data, graph, visuals, metrics

# Fetch and build graph
prices = data.fetch_stock_data(['RELIANCE.NS', 'TCS.NS', 'INFY.NS'],
                               '2023-01-01', '2024-01-01')
G, corr = graph.build_graph_from_prices(prices, threshold=0.5)

# Color nodes by degree centrality
degree_cent = metrics.compute_degree_centrality(G)

# Create custom visualization
fig = visuals.create_network_plot(
    G,
    node_colors=degree_cent,
    node_size_metric=degree_cent,
    layout='kamada_kawai',
    title='Stock Network - Colored by Degree Centrality'
)
fig.show()
```

## Nifty 500 Index Data

Download the Nifty 500 index data:

```bash
# Windows (PowerShell)
Invoke-WebRequest -Uri https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv -OutFile res/index_nifty500.csv

# Linux / macOS
wget -O res/index_nifty500.csv https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Stock data provided by [Yahoo Finance](https://finance.yahoo.com/) via yfinance
- Network analysis powered by [NetworkX](https://networkx.org/)
- Visualizations created with [Plotly](https://plotly.com/)
- Web interface built with [Dash](https://dash.plotly.com/)

## References

Reference papers on stock market network analysis are available in the `papers/` directory.

## Contact

**Author**: Adheeesh Trivedi
**GitHub**: [@adhxtry](https://github.com/adhxtry)
**Repository**: [nifty-network-analysis](https://github.com/adhxtry/nifty-network-analysis)

[1] Chi, K. Tse, Jing Liu, and Francis CM Lau. "A network perspective of the stock market." Journal of
Empirical Finance 17.4 (2010): 659-667.

[2] Moghadam, Hadi Esmaeilpour, et al. "Complex networks analysis in Iran stock market: The
application of centrality." Physica A: Statistical Mechanics and its Applications 531 (2019): 121800.

They are also included in the [papers](papers/) folder.