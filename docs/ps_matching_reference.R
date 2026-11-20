# Authored-but-not-executed R reference.
# This project was built and validated in Python (see src/) because R is
# not available in this execution environment. This file documents how the
# same real-world-evidence causal pipeline would be written in R using the
# standard MatchIt + survival stack, for reference/portability.

library(MatchIt)
library(survival)
library(cobalt)

df <- read.csv("simulated_patients.csv")

# Propensity score estimation + 1:1 nearest-neighbor matching with caliper
m.out <- matchit(
  treatment ~ age + severity + comorbidity_score + sex,
  data = df,
  method = "nearest",
  distance = "glm",
  caliper = 0.05
)

matched_data <- match.data(m.out)

# Balance diagnostics (standardized mean differences)
bal.tab(m.out, un = TRUE)

# Outcome model on matched data: Cox proportional hazards
cox_matched <- coxph(Surv(time, event) ~ treatment, data = matched_data)
summary(cox_matched)

# IPTW using the same propensity model
df$ps <- m.out$distance
df$iptw <- ifelse(df$treatment == 1, 1 / df$ps, 1 / (1 - df$ps))
cox_iptw <- coxph(Surv(time, event) ~ treatment, data = df, weights = iptw, robust = TRUE)
summary(cox_iptw)

# Stratification on propensity-score quintiles
df$stratum <- cut(df$ps, breaks = quantile(df$ps, probs = seq(0, 1, 0.2)), include.lowest = TRUE)
cox_strat <- coxph(Surv(time, event) ~ treatment + strata(stratum), data = df)
summary(cox_strat)
