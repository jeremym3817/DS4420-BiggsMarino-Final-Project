# Bayesian Model Plan
# Project: Batting vs Pitching Relationship in College Baseball

# 1. Idea:
# We want to see how batting and pitching relate to team success across multiple seasons of college baseball.
#
# The data only has end-of-season stats for each team.
# Teams are also grouped into conferences.
#
# Because the data spans multiple years (2012-2025), we want to see how the
# batting vs pitching relationship changes over time.

# 2. What the data looks like
#
# Years are strange since some data doesn't exist until a certain year. For example, OBP starts in 2012, while SBPG starts in 2008.
# So the year will be decided after picking all the stats we need. We will likely have to start in 2012 to get all the stats we want. 
# Also the 2026 is ongoing, so might scrape that right before presentation.
#
# Each row will represent one team in one season.
#
# Example columns:
#
# team
# conference
# season
# batting_stat
# pitching_stat
# team_success (W or WPCT or run differential (R (Batting) - R (Pitching)))
#
# Batting stat could be something like:
# OPS or RPG
#
# Pitching stat could be something like:
# ERA or R (Pitching) (Runs Allowed Pitched)

# 3. Basic idea of the model
#
# Basic relationship:
#
# team_success =
#   intercept
#   + batting_effect * batting_stat
#   + pitching_effect * pitching_stat
#
# Bayesian models estimate probability distributions for these effects.

# 4. Allow the relationship to change by season
#
# The importance of batting and pitching might change every year. So the model will estimate separate effects for each season.
#
# Example:
#
# batting_effect_2019
# batting_effect_2020
# batting_effect_2021
#
# pitching_effect_2019
# pitching_effect_2020
# pitching_effect_2021
#
# This lets us see if batting changes over time.

# 5. Conferences
#
# Teams belong to conferences and conferences may differ in
# overall strength.
#
# Could use RPI - Conference Rankings?

# 6. Priors
#
# Example priors:
#
# intercept ~ Normal()
# batting_effect ~ Normal()
# pitching_effect ~ Normal()
#
# These are just starting assumptions and the data should update them.

# 7. What we want from the model
# - how strong the batting effect is each season
# - how strong the pitching effect is each season
#
# This tells us whether college baseball has become more offense-driven or pitching-driven over time.

# 8. Steps to implement
#
# a. Load and clean the data
#
# b. Choose batting and pitching metrics
#
# c. Standardize the stats if needed
#
# d. Encode teams, conferences, and seasons
#
# e. Fit Bayesian model
#
# f. Extract the batting and pitching effects for each season



library(jsonlite)
library(ggplot2)
library(dplyr)

data <- fromJSON("data/div1/2012.json")
data_sorted <- data[order(names(data))]

# interactive viewer in RStudio
str(data_sorted, max.level = 2)

# Now moving over to all data
years <- 2012:2025
all_rows <- list()
row_i <- 1

for (yr in years) {
  path <- paste0("data/div1/", yr, ".json")
  if (!file.exists(path)) next

  raw <- fromJSON(path)

  # Each element is a team; convert to rows
  for (team_key in names(raw)) {
    team_info <- as.data.frame(raw[[team_key]], stringsAsFactors = FALSE)
    team_info$team_key <- team_key
    team_info$season <- yr
    all_rows[[row_i]] <- team_info
    row_i <- row_i + 1
  }
}

all_data <- bind_rows(all_rows)

model_data <- data.frame(
  team = all_data$team,
  conference = all_data$league,
  season = all_data$season,
  wpct = as.numeric(all_data$WPCT),
  obp = as.numeric(all_data$OBP),
  rpg = as.numeric(all_data$RPG),
  era = as.numeric(all_data$ERA),
  r_pitching = as.numeric(all_data$R..Pitching.),
  r_batting = as.numeric(all_data$R..Batting.),
  slg = as.numeric(all_data$SLG),
  g = as.numeric(all_data$G),
  stringsAsFactors = FALSE
)

model_data$ops <- model_data$slg + model_data$obp   # OPS = OBP + SLG (since raw OPS isn't in the data)
model_data$run_diff <- model_data$r_batting - model_data$r_pitching

model_data$ops_z <- (model_data$ops - mean(model_data$ops, na.rm = TRUE)) / sd(model_data$ops, na.rm = TRUE)
model_data$era_z <- (model_data$era - mean(model_data$era, na.rm = TRUE)) / sd(model_data$era, na.rm = TRUE)

model_data <- model_data[complete.cases(model_data$wpct, model_data$ops_z, model_data$era_z), ]

# ---------- manual bayesian model using metropolis-hastings ----------

unique_seasons <- sort(unique(model_data$season))
unique_confs <- sort(unique(model_data$conference))
n_seasons <- length(unique_seasons)
n_confs <- length(unique_confs)

# number of params: intercept + batting effect per season + pitching effect per season + conference random effects + sigma
n_params <- 1 + n_seasons + n_seasons + n_confs + 1

# log likelihood function
log_likelihood <- function(params, data) {
  intercept <- params[1]
  beta_bat <- params[2:(1 + n_seasons)]
  beta_pit <- params[(2 + n_seasons):(1 + 2 * n_seasons)]
  conf_effects <- params[(2 + 2 * n_seasons):(1 + 2 * n_seasons + n_confs)]
  sigma <- exp(params[length(params)])  # exp so it stays positive

  ll <- 0
  for (i in 1:nrow(data)) {
    s_idx <- which(unique_seasons == data$season[i])
    c_idx <- which(unique_confs == data$conference[i])

    mu <- intercept + beta_bat[s_idx] * data$ops_z[i] + beta_pit[s_idx] * data$era_z[i] + conf_effects[c_idx]
    ll <- ll + dnorm(data$wpct[i], mean = mu, sd = sigma, log = TRUE)
  }
  return(ll)
}

# log prior
log_prior <- function(params) {
  intercept <- params[1]
  beta_bat <- params[2:(1 + n_seasons)]
  beta_pit <- params[(2 + n_seasons):(1 + 2 * n_seasons)]
  conf_effects <- params[(2 + 2 * n_seasons):(1 + 2 * n_seasons + n_confs)]
  log_sigma <- params[length(params)]

  lp <- dnorm(intercept, mean = 0.5, sd = 0.2, log = TRUE)
  lp <- lp + sum(dnorm(beta_bat, mean = 0, sd = 0.5, log = TRUE))
  lp <- lp + sum(dnorm(beta_pit, mean = 0, sd = 0.5, log = TRUE))
  lp <- lp + sum(dnorm(conf_effects, mean = 0, sd = 0.1, log = TRUE))
  lp <- lp + dnorm(log_sigma, mean = log(0.1), sd = 1, log = TRUE)

  return(lp)
}

# log posterior = log likelihood + log prior
log_posterior <- function(params, data) {
  return(log_likelihood(params, data) + log_prior(params))
}

# metropolis-hastings sampler
run_mcmc_block <- function(data, n_iter = 20000, burn_in = 5000) {
  
  # initialize params
  current <- rep(0, n_params)
  current[1] <- 0.5
  current[length(current)] <- log(0.1)
  
  # define blocks and their proposal SDs
  idx_intercept <- 1
  idx_bat <- 2:(1 + n_seasons)
  idx_pit <- (2 + n_seasons):(1 + 2 * n_seasons)
  idx_conf <- (2 + 2 * n_seasons):(1 + 2 * n_seasons + n_confs)
  idx_sigma <- length(current)
  
  blocks <- list(idx_intercept, idx_bat, idx_pit, idx_conf, idx_sigma)
  block_sd <- c(0.005, 0.002, 0.002, 0.002, 0.05)
  block_accept <- rep(0, length(blocks))
  block_total <- rep(0, length(blocks))
  
  samples <- matrix(NA, nrow = n_iter, ncol = n_params)
  current_lp <- log_posterior(current, data)
  
  cat("Running block MCMC...\n")
  for (iter in 1:n_iter) {
    
    # cycle through each block
    for (b in seq_along(blocks)) {
      proposal <- current
      idx <- blocks[[b]]
      proposal[idx] <- rnorm(length(idx), mean = current[idx], sd = block_sd[b])
      
      proposed_lp <- log_posterior(proposal, data)
      log_ratio <- proposed_lp - current_lp
      
      block_total[b] <- block_total[b] + 1
      if (log(runif(1)) < log_ratio) {
        current <- proposal
        current_lp <- proposed_lp
        block_accept[b] <- block_accept[b] + 1
      }
    }
    
    samples[iter, ] <- current
    
    if (iter %% 2000 == 0) {
      rates <- round(block_accept / block_total, 3)
      cat("Iteration", iter, "- block acceptance rates:",
          "intercept:", rates[1],
          "bat:", rates[2],
          "pit:", rates[3],
          "conf:", rates[4],
          "sigma:", rates[5], "\n")
    }
  }
  
  cat("Done!\n")
  samples <- samples[(burn_in + 1):n_iter, ]
  return(samples)
}

set.seed(42)
samples <- run_mcmc_block(model_data, n_iter = 20000, burn_in = 5000)

# Pull posterior draws for the season-specific effects
intercept_draws <- samples[, 1]
cat("Intercept mean:", mean(intercept_draws), "\n")
cat("Intercept 95% CI:", quantile(intercept_draws, c(0.025, 0.975)), "\n")

# Or more conveniently:
bat_means <- c()
bat_lo <- c()
bat_hi <- c()
pit_means <- c()
pit_lo <- c()
pit_hi <- c()

for (s in 1:n_seasons) {
  bat_col <- 1 + s
  pit_col <- 1 + n_seasons + s

  bat_means[s] <- mean(samples[, bat_col])
  bat_lo[s] <- quantile(samples[, bat_col], 0.025)
  bat_hi[s] <- quantile(samples[, bat_col], 0.975)

  pit_means[s] <- mean(samples[, pit_col])
  pit_lo[s] <- quantile(samples[, pit_col], 0.025)
  pit_hi[s] <- quantile(samples[, pit_col], 0.975)
}

# For a custom plot of how effects evolve:
plot_df <- data.frame(
  year = rep(unique_seasons, 2),
  effect = c(bat_means, pit_means),
  lo = c(bat_lo, pit_lo),
  hi = c(bat_hi, pit_hi),
  type = rep(c("Batting (OPS)", "Pitching (ERA)"), each = n_seasons)
)

ggplot(plot_df, aes(x = year, y = effect, color = type)) +
  geom_line() +
  geom_ribbon(aes(ymin = lo, ymax = hi, fill = type), alpha = 0.2) +
  labs(y = "Effect on WPCT", title = "Batting vs Pitching Effect Over Time") +
  theme_minimal()

