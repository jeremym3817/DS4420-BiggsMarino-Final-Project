# BATTING AND PITCHING METRIC’S EVOLUTION IN COLLEGE BASEBALL: A TIME SERIES AND BAYESIAN ANALYSIS

## Authors
Mateo Biggs and Jeremy Marino (DS4420 — Machine Learning & Data Mining, Northeastern University, Spring 2026)

## Overview
This project investigates the evolving relationship between batting and pitching performance across NCAA Division I, II, and III college baseball from 2012 to 2025. Using a combination of **time series models** and **Bayesian statistical methods**, we analyze how key offensive and defensive metrics have trended over the past 14 seasons, forecast where they're heading, and quantify their impact on team success.

## Research Questions
- How have pitching and batting metrics evolved in at the collegiate level?
- Are these changes uniform amongst all divisions?
- How can we quantify how uncertain we are of these future outcomes?
- Can we effectively project how these metrics will change moving forward?

## Data
Season-level team statistics scraped across all three NCAA divisions (2012–2025), organized as JSON files per year per division.

**Batting Metrics:** BA (Batting Average), OBP (On-Base Percentage), SLG (Slugging Percentage), HRPG (Home Runs Per Game), RPG (Runs Per Game)

**Pitching Metrics:** ERA (Earned Run Average), WHIP (Walks + Hits Per Inning Pitched), K/9 (Strikeouts Per 9 Innings), K/BB (Strikeout-to-Walk Ratio)

*Note: OBP, WHIP, and K/BB are derived from raw counting stats (H, BB, HBP, SF, AB, HA, IP, SO) using standard formulas.*

### Data Structure
```
Data/
├── div1/
│   ├── 2012.json
│   ├── 2013.json
│   └── ...2025.json
├── div2/
│   └── ...
└── div3/
    └── ...
```

Each JSON file contains team-level season statistics with conference affiliations. Data is aggregated to conference-level averages for time series modeling and used at the team level for Bayesian analysis.

## Methodology

### Time Series Analysis

**Panel Autoregression (AR):** A from-scratch OLS implementation of autoregressive models using a panel data structure. All conferences within a division are stacked together (~30 conferences × 12 years ≈ 360 observations), with conference dummy variables to control for baseline differences. A separate division-average AR model with a linear trend term handles forecasting.

- ACF/PACF analysis guides lag selection
- AR(1) selected for most metrics based on PACF cutoff
- Linear trend term included for metrics with clear directional movement

**ARIMA/MA Models:** **TBD** (model was not finished in time, for future iterations)

### Bayesian Analysis


## Installation & Requirements
```bash
pip install numpy pandas matplotlib statsmodels pymc arviz
```

**Core dependencies:**
- `numpy` — matrix operations, OLS normal equation
- `pandas` — data manipulation and panel construction
- `matplotlib` — visualization
- `statsmodels` — ADF tests, ACF/PACF, ARIMA validation
- `pymc` — Bayesian hierarchical modeling

## Usage

### Load and Preprocess Data
```python
# Build conference-level historical data
conf_dict = get_historic_conference_data(div=1)

# Aggregate derived metrics (OBP, WHIP, K/BB)
df = aggregate_metrics(df)
```

### Time Series Analysis (Example Functions)
```python
# Visualize conference trends
get_conf_trend(conf_dict, metric, conf_=None, all_conf=False, div=1, visualizations=False)

# ACF/PACF for lag selection
get_acf_pacf(metric='ERA', metric_df=avg_metric_df, div=1)

# Fit panel AR model with forecast
panel, coeffs, avg_coeffs, results, summary = panel_arpanel_ar(
    df, metric, div=1, lag=1, test_years=2, forecast_years=3, display_stats=True, visualizations=False
)
```
