# Strategies Overview

This folder contains 16 statistical/probabilistic strategies integrated with `app_advanced.py`.  
Each strategy is implemented as a class with `__init__(df, **params)` and `run()` methods, returning a list of trade signals.

---

### 1. sigma_extreme.py — **Sigma Extreme**
Enters trades when returns exceed a specified number of standard deviations (σ) from the mean.  
Probabilistic edge: reversion after extreme outliers.

---

### 2. opening_range_breakout.py — **Opening Range Breakout**
Defines the high/low of the first X minutes after market open and trades a breakout beyond those levels.  
Probabilistic edge: volatility clustering at session start.

---

### 3. gap_fade.py — **Gap Fade / Continuation**
Looks at overnight gaps (close-to-open). Trades fade if gap is extreme, continuation otherwise.  
Probabilistic edge: mean reversion vs. momentum depending on gap size.

---

### 4. time_of_day_mean_reversion.py — **Time-of-Day Mean Reversion**
Exploits recurring intraday tendencies (e.g., reversal around NY lunch hour).  
Probabilistic edge: conditional expectation on time-of-day.

---

### 5. microstructure_imbalance.py — **Microstructure Imbalance**
Uses imbalance between up-ticks and down-ticks, or volume imbalance, to generate signals.  
Probabilistic edge: order flow imbalance predicts short-term moves.

---

### 6. mean_reversion_returns.py — **Mean Reversion on Returns**
Trades when short-term returns deviate significantly from the average.  
Probabilistic edge: short-term reversals after extreme price changes.

---

### 7. prob_time_of_day.py — **Probabilistic Time-of-Day Edge**
Estimates probability of up vs. down moves at different times of day.  
Probabilistic edge: statistical bias conditional on session hour.

---

### 8. overnight_gap.py — **Overnight Gap Strategy**
Focuses only on open-to-close performance after overnight gaps.  
Probabilistic edge: gap direction is predictive for early session.

---

### 9. prob_sign_sequences.py — **Probability of Sign Sequences**
Analyzes sequences of consecutive up/down bars and trades reversal or continuation.  
Probabilistic edge: run-length distribution deviates from randomness.

---

### 10. return_clustering.py — **Return Clustering (GARCH-like)**
Models conditional variance (volatility clustering) and trades when volatility is predicted to rise/fall.  
Probabilistic edge: volatility tends to cluster in time.

---

### 11. extreme_quantile.py — **Extreme Quantile Strategy**
Triggers when returns are in the extreme tails (quantiles) of distribution.  
Probabilistic edge: fat-tailed return distribution.

---

### 12. expected_value_max.py — **Expected Value Maximization**
Optimizes entry based on expected value of conditional outcomes.  
Probabilistic edge: positive expectancy estimation.

---

### 13. sequential_reversal.py — **Sequential Reversal / Run-Length**
Trades reversal after extended streaks of bars in one direction.  
Probabilistic edge: streaks become statistically unlikely.

---

### 14. monte_carlo_bootstrap.py — **Monte Carlo / Bootstrap**
Resamples historical returns to simulate distributions and generate probabilistic thresholds.  
Probabilistic edge: resampled expected outcomes vs. current returns.

---

### 15. volatility_breakout.py — **Volatility Breakout**
Enters when price moves beyond a multiple of ATR (or std dev) from recent range.  
Probabilistic edge: volatility expansion often continues.

---

### 16. mean_reversion_price_changes.py — **Mean Reversion on Price Changes**
Focuses directly on raw price changes (not returns). Large moves tend to revert.  
Probabilistic edge: reversal after abnormal price shifts.

---

## Usage
Each strategy:
- Inherits from a simple `StrategyBase` (or standalone class).
- Has parameters configurable via `app_advanced.py`.
- Returns signals in format:  
  ```python
  [(datetime, "LONG" or "SHORT", price), ...]
