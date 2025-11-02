# Nifty 500 Stock Market Network Analysis

A webapp for Nifty 500 Stock Market Network Analysis.
It includes analysis of stock market data using network science techniques.

### Setup

#### Clone the repository:

```bash
git clone https://github.com/adhxtry/nifty-network-analysis.git
cd nifty-network-analysis
```

#### Install dependencies:

1. Using [uv](https://docs.astral.sh/uv/)

```bash
uv install
```

2. Using `pip`

```bash
pip install -r requirements.txt
```

### Nifty 500 index

Download the Nifty 500 index data from [Nifty indices](https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv).
Put the CSV file in the `res/` folder.

Or simply run:

```bash
# Linux / MacOS
wget -O res/index_nifty500.csv https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv
# Powershell
Invoke-WebRequest -Uri https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv -OutFile res/index_nifty500.csv
```

### Historical stock data

Fetch the

## References

Reference papers on stock market network analysis are listed below:

[1] Chi, K. Tse, Jing Liu, and Francis CM Lau. "A network perspective of the stock market." Journal of
Empirical Finance 17.4 (2010): 659-667.

[2] Moghadam, Hadi Esmaeilpour, et al. "Complex networks analysis in Iran stock market: The
application of centrality." Physica A: Statistical Mechanics and its Applications 531 (2019): 121800.

They are also included in the [papers](papers/) folder.