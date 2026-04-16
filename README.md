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

**Hierarchical Linear Regression via Metropolis-Hastings:** A from-scratch implementation in R estimating the effect of run production and run prevention on win percentage. All three divisions are pooled at the team level (~12,000 team-season observations), with season-varying coefficients for R (Batting) and R (Pitching) and conference-level random effects to capture structural differences across ~130 conferences.

- Weakly informative priors: intercept centered at .500, moderate effect size priors on slopes, tight prior on conference effects
- Proposal standard deviations hand-tuned to achieve 25–35% acceptance rate
- 10,000 iterations with 5,000 burn-in
- 2020 season excluded due to COVID

**Core R dependencies:**
- `jsonlite` — JSON parsing for season data files
- `dplyr` — data manipulation and cleaning
- `ggplot2` — posterior visualization

### Bayesian Analysis (Example Usage)
```r
# Load and combine all divisions (2011–2025, excluding 2020)
all_data <- bind_rows(all_rows)

# Z-score predictors
model_data$rbat_z <- (model_data$r_batting - mean(model_data$r_batting, na.rm = TRUE)) / sd(model_data$r_batting, na.rm = TRUE)
model_data$rpit_z <- (model_data$r_pitching - mean(model_data$r_pitching, na.rm = TRUE)) / sd(model_data$r_pitching, na.rm = TRUE)

# Run MCMC sampler
set.seed(42)
samples <- run_mcmc(model_data, n_iter = 10000, burn_in = 5000)

# Extract season-varying effects and plot
ggplot(plot_df, aes(x = year, y = effect, color = type)) +
  geom_line() +
  geom_ribbon(aes(ymin = lo, ymax = hi, fill = type), alpha = 0.2) +
  theme_minimal()
```

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
