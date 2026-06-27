# regime-momentum-signal
Volatility regime detection and momentum signal backtest on SPY/VIX

## Motivation

Build a systematic signal that adapts to volatility regimes, as a way to develop hands-on research and backtesting
skills beyond theoretical risk modeling.

## Data

SPY + VIX, daily, 2010-2024, via yfinance. Using Close prices (not dividend-adjusted) — a simplification to revisit if
returns need to reflect total return.

## Methodology

**Regime detection:** short-term volatility (20-day rolling) compared against long-term volatility. Two long-window
specifications were evaluated: 60-day (one quarter) and 252-day (one trading year). The 252-day specification was
selected as it produced more persistent regimes and more stable switch counts across threshold values — the 20d/60d
specification showed a sharp sensitivity to threshold choice (186 switches at 1.1 vs 45 at 1.4), while the 20d/252d
specification was more robust (106 at 1.1 vs 47 at 1.4).

**Parameter robustness:** thresholds from 1.1 to 1.4 were tested on both window pairs. The 20d/252d specification produced
stable switch counts across 1.2-1.4 (55-61 switches over 14 years, approximately 4 per year), suggesting results are not
sensitive to the exact threshold in this range. A threshold of 1.2 was selected. Regime persistence of approximately 3
months on average is consistent with the momentum holding periods documented in Jegadeesh & Titman (1993).

## Signals

**Regime signal:** `regime = +1` when `rolling_20 < rolling_252 * 1.2` (low vol), `-1` otherwise (high vol).

**Momentum signal:** (to be added — 20-day return, combined with regime)

## Backtest

## Results

## Limitations
- Past performance doesn't guarantee future regimes/results 
- Using raw Close, not dividend-adjusted — understates true returns
- No transaction costs yet (to be added once a backtest exists)
- In-sample only so far — no train/test split yet

