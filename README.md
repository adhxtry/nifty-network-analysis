# Nifty Network Analysis

A comprehensive Python project for analyzing and visualizing network relationships between Nifty companies using stock market data.

## Demo

Here is a live demo of the web app:

![Live Demo Video](docs/demo.gif)

For the more detailed usage, refer to the [Prototype Notebook](./prototype.ipynb) which shows step-by-step analysis and usage of the library.

## Features

- **Stock Data Fetching**: Fetch historical stock data using yfinance with built-in caching
- **Network Analysis**: Build correlation-based graphs using NetworkX
- **Power Law Analysis**: Test for scale-free network characteristics across multiple thresholds
- **Centrality Metrics**: Compute degree, betweenness, closeness, eigenvector centrality, and PageRank
- **Community Detection**: Identify clusters of correlated stocks
- **Interactive Visualizations**: Create beautiful plots with Plotly
- **Web Application**: Full-featured Dash web interface for network analysis
- **Download Results**: Export all plots (PNG) and analysis reports (TXT) as ZIP files
- **Documentation**: Complete API documentation with Sphinx

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
│   ├── config.py         # Configuration settings
│   ├── requirements.txt  # Web app dependencies
│   └── README.md         # Web app documentation
├── tests/                 # Unit tests
├── docs/                  # Sphinx documentation (TODO)
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

### Install Notebook Dependencies

To run [prototype notebook](./prototype.ipynb), install additional dependencies:

```bash
uv sync --extra notebook
```

## Quick Start

### Using the Web Application

The web application provides a complete interface for network analysis with interactive visualizations and downloadable results.

#### Using uv (Recommended)

```bash
# Run the web app
uv run webapp/app.py
```

#### Using pip

```bash
# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Run the web app
python webapp/app.py
```

Then open your browser to: http://localhost:8050

#### Using the Web App

**Step 1: Data Configuration**
- Enter the NSE index CSV URL (default: Nifty Total Market)
- Select date range for historical data (e.g., 2024-01-01 to 2025-10-01)
- Set threshold range for power law analysis (e.g., 0.2 to 0.7)
- Check "Force Fetch" to bypass cache and get fresh data
- Click **"Fetch Data & Generate Threshold Analysis"**

**Step 2: Power Law Analysis**
- View degree distribution plots on log-log scale for multiple thresholds
- Review network statistics table (nodes, edges, average degree, R² values)
- Download results as ZIP (plot PNG + stats TXT)

**Step 3: Detailed Network Analysis**
- Select a specific correlation threshold
- Click **"Analyze"** to generate comprehensive analysis
- View:
  - Network statistics overview
  - Most/least correlated stock pairs
  - 5 centrality measures with interactive visualizations
  - Community structure (clusters of correlated stocks)
  - Clustering coefficients
- Download all results as ZIP (5 centrality plots PNG + complete report TXT)

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

### Python Library API

The `niftynet` library provides the core functionality used by the web application.

## Data Sources

The web application supports fetching stock data from NSE (National Stock Exchange of India):

**Default Index URL:**
```
https://nsearchives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv
```

**Alternative Index URLs:**
- Nifty 500: `https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv`
- Nifty 50: `https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv`

Stock price data is fetched automatically using yfinance and cached locally in `~/.niftynet/data/` for faster subsequent runs.

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
- NSE index data from [NSE India](https://www.nseindia.com/)
- Network analysis powered by [NetworkX](https://networkx.org/)
- Visualizations created with [Plotly](https://plotly.com/)
- Web interface built with [Dash](https://dash.plotly.com/)
- Image export using [Kaleido](https://github.com/plotly/Kaleido)

## Contact

**Author**: Adheeesh Trivedi
**GitHub**: [@adhxtry](https://github.com/adhxtry)
**Repository**: [nifty-network-analysis](https://github.com/adhxtry/nifty-network-analysis)

## References

Reference papers on stock market network analysis are available in the `papers/` directory.

[1] Chi, K. Tse, Jing Liu, and Francis CM Lau. "A network perspective of the stock market." Journal of
Empirical Finance 17.4 (2010): 659-667.

[2] Moghadam, Hadi Esmaeilpour, et al. "Complex networks analysis in Iran stock market: The
application of centrality." Physica A: Statistical Mechanics and its Applications 531 (2019): 121800.