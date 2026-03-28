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
library(tidyverse)
install.packages("listviewer")

data <- fromJSON("data/div1/2012.json")
data_sorted <- data[order(names(data))]

# interactive viewer in RStudio
listviewer::jsonedit(data_sorted)

# Now moving over to all data
years <- 2012:2025
all_data <- map_dfr(years, function(yr) {
  path <- paste0("data/div1/", yr, ".json")
  if (!file.exists(path)) return(NULL)
  
  raw <- fromJSON(path)
  
  # Each element is a team; convert to rows
  map_dfr(raw, ~as_tibble(.x), .id = "team_key") %>%
    mutate(season = yr)
})

model_data <- all_data %>%
  select(
    team,
    conference = league,
    season,
    wpct = WPCT,
    ops = OBP,         # or use OPS if you compute it: SLG + OBP
    rpg = RPG,
    era = ERA,
    r_pitching = `R (Pitching)`,
    r_batting = `R (Batting)`,
    slg = SLG,
    g = G
  ) %>%
  mutate(
    ops = slg + ops,   # OPS = OBP + SLG (since raw OPS isn't in the data)
    run_diff = r_batting - r_pitching,
    season_f = factor(season)
  )

model_data <- model_data %>%
  mutate(
    ops_z = (ops - mean(ops, na.rm = TRUE)) / sd(ops, na.rm = TRUE),
    era_z = (era - mean(era, na.rm = TRUE)) / sd(era, na.rm = TRUE)
  )

install.packages("brms")
library(brms)

fit <- brm(
  wpct ~ 1 + ops_z:season_f + era_z:season_f + (1 | conference),
  data = model_data,
  family = gaussian(),
  prior = c(
    prior(normal(0.5, 0.2), class = "Intercept"),
    prior(normal(0, 0.5), class = "b"),
    prior(normal(0, 0.1), class = "sd")
  ),
  chains = 4,
  iter = 2000,
  cores = 4
)

# Pull posterior draws for the season-specific effects
draws <- as_draws_df(fit)

# Or more conveniently:
conditional_effects(fit, effects = "ops_z:season_f")

# For a custom plot of how effects evolve:
posterior_summary(fit) %>%
  as.data.frame() %>%
  rownames_to_column("param") %>%
  filter(str_detect(param, "ops_z|era_z")) %>%
  mutate(
    type = if_else(str_detect(param, "ops"), "Batting (OPS)", "Pitching (ERA)"),
    year = as.numeric(str_extract(param, "\\d{4}"))
  ) %>%
  ggplot(aes(x = year, y = Estimate, color = type)) +
  geom_line() +
  geom_ribbon(aes(ymin = Q2.5, ymax = Q97.5, fill = type), alpha = 0.2) +
  labs(y = "Effect on WPCT", title = "Batting vs Pitching Effect Over Time")

