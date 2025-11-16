"""
Configuration file for the Nifty Network Analysis web application.

Customize these settings to change the app's behavior.
"""

# Server Configuration
SERVER_HOST = 'localhost'
SERVER_PORT = 8050
DEBUG_MODE = False

# Default Values
DEFAULT_INDEX_URL = "https://nsearchives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv"
DEFAULT_START_DATE = "2024-01-01"
DEFAULT_END_DATE = "2025-10-01"
DEFAULT_THRESHOLD_MIN = 0.2
DEFAULT_THRESHOLD_MAX = 0.7
DEFAULT_THRESHOLD_STEP = 0.05
DEFAULT_ANALYSIS_THRESHOLD = 0.4

# Analysis Parameters
TOP_N_CENTRALITY = 15
TOP_N_CORRELATION = 10
TOP_N_COMMUNITIES = 10

# UI Theme
# Available themes: BOOTSTRAP, CERULEAN, COSMO, CYBORG, DARKLY, FLATLY, JOURNAL,
# LITERA, LUMEN, LUX, MATERIA, MINTY, MORPH, PULSE, QUARTZ, SANDSTONE, SIMPLEX,
# SKETCHY, SLATE, SOLAR, SPACELAB, SUPERHERO, UNITED, VAPOR, YETI, ZEPHYR
BOOTSTRAP_THEME = 'DARKLY'

# Plot Configuration
PLOT_HEIGHT_MULTIPLIER = 400  # Height per row in threshold analysis
CENTRALITY_PLOT_HEIGHT = 600  # Base height for centrality plots

# Cache Configuration (uses niftynet defaults)
# These are inherited from the niftynet library
# To change caching behavior, modify ~/.niftynet/config.json
