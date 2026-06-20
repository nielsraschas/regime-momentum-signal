# regime-momentum-signal
Volatility regime detection and momentum signal backtest on SPY/VIX

## Motivation

Build a systematic signal that adapts to volatility regimes, as a way to develop hands-on research and backtesting
skills beyond theoretical risk modeling.

## Data

SPY + VIX, daily, 2010-2024, via yfinance. Using Close prices (not dividend-adjusted) — a simplification to revisit if
returns need to reflect total return.

## Methodology
Rolling volatility regime detection (20d/60d), to be followed by a momentum signal.

## Signals

## Backtest

## Results

## Limitations
- Past performance doesn't guarantee future regimes/results 
- Using raw Close, not dividend-adjusted — understates true returns
- No transaction costs yet (to be added once a backtest exists)
- In-sample only so far — no train/test split yet

