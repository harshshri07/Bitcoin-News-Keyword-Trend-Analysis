# Bitcoin News Keyword Trend Analysis API Documentation

## Overview
This project analyzes trends in Bitcoin-related news and correlates them with Bitcoin price movements. The API layer provides a clean, reusable interface for fetching news and price data, extracting keywords, analyzing trends, performing Granger causality testing, visualizing results, and generating a dashboard. All logic is encapsulated in `textblob1_utils.py` for use in notebooks and other applications.

## Native API Description
### NewsAPI
- **Endpoint:** `https://newsapi.org/v2/everything`
- **Purpose:** Fetches news articles containing the keyword "bitcoin".
- **Authentication:** Requires an API key (passed as `apiKey` parameter).
- **Parameters:**
  - `q`: Query string (e.g., 'bitcoin')
  - `language`: 'en'
  - `sortBy`: 'publishedAt'
  - `pageSize`: Up to 100
  - `page`: Pagination
- **Rate Limits:** 100 requests/day (free tier)
- **Response Schema:**
  - `title`, `description`, `publishedAt`, `source` (name)

### Price Data
- **Source:** (Sample data in this project; can be replaced with real API)
- **Schema:**
  - `timestamp`, `price`, `date`, `hour`

## Wrapper Layer Design
All API and processing logic is wrapped in functions in `textblob1_utils.py`:

### Data Fetching
- `fetch_bitcoin_news(api_key=None, days=30)`: Fetches news from NewsAPI or returns sample data.
- `fetch_bitcoin_prices(days=30, interval='daily')`: Returns sample price data.
- `fetch_current_bitcoin_price()`: Returns the current Bitcoin price.

### Data Processing
- `extract_keywords(news_df=None, min_frequency=1, include_titles=True)`: Extracts keywords from news using TextBlob with enhanced filtering.
- `analyze_trends(news_with_keywords=None, price_df=None, time_window='daily', min_freq=1)`: Merges keyword and price data, computes trends and correlations.

### Statistical Analysis
- `run_granger_tests(merged_df=None, top_n=10, max_lag=3, min_data_points=10)`: Performs Granger causality tests to determine if keywords can predict price changes or vice versa.
- `run_stationarity_test(series)`: Tests if a time series is stationary using the Augmented Dickey-Fuller test.
- `make_stationary(series)`: Transforms a time series to make it stationary.

### Visualization
- `plot_keyword_vs_price(merged_df, keyword, save_fig=False, display_fig=True)`: Plots keyword frequency vs price.
- `plot_keyword_price_heatmap(merged_df=None, top_n=10, save_fig=False, display_fig=True)`: Plots heatmap of keyword-price correlations.

### Dashboard Generation
- `generate_dashboard(merged_df=None, output_path='dashboard/index.html')`: Generates a comprehensive HTML dashboard.

### Utility Functions
- `is_docker()`: Checks if the code is running inside a Docker container.
- `is_good_keyword(kw)`: Determines if a keyword is valuable for analysis.
- `extract_important_keywords(text, is_title=False)`: Extracts and weights keywords intelligently.

## Example Usage
```python
from textblob1_utils import *

# Fetch data
news_df = fetch_bitcoin_news(api_key=None, days=7)
price_df = fetch_bitcoin_prices(days=7)
current_price = fetch_current_bitcoin_price()

# Process data
news_with_keywords = extract_keywords(news_df)
merged_df = analyze_trends(news_with_keywords, price_df)

# Perform statistical analysis
causality_results = run_granger_tests(merged_df, top_n=5, max_lag=2, min_data_points=5)

# Visualize results
plot_keyword_vs_price(merged_df, 'bitcoin')
plot_keyword_price_heatmap(merged_df)

# Generate dashboard
dashboard_path = generate_dashboard(merged_df)
```

## Design Rationale
- **Separation of Concerns:** Each function does one job (fetch, process, plot, etc.)
- **Reusability:** All logic is in `textblob1_utils.py` for easy use in notebooks and scripts.
- **Fallbacks:** Sample data is used if no API key is provided, ensuring the project runs end-to-end for all users.
- **Extensibility:** Functions can be swapped for real APIs or extended for more features.
- **Notebook Simplicity:** Notebooks remain clean, with all logic abstracted away in the utility module.
- **Smart Functionality:** Files and data are automatically saved, allowing functions to reload from disk when needed.
- **Statistical Rigor:** Granger causality testing helps validate potential predictive relationships between keywords and price movements. 