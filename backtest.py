import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def give_regime_data():
    spy = pd.read_csv("data/spy.csv", parse_dates=["Date"])
    spy["returns"] = spy["Close"].pct_change()
    spy["rolling_20"] = spy["returns"].rolling(20).std() * (252)**(1/2)
    spy["rolling_252"] = spy["returns"].rolling(252).std() * (252) ** (1 / 2)
    spy["regime"] = 1
    spy.loc[spy["rolling_20"] >= spy["rolling_252"]*1.2, "regime"] = 0
    return spy


if __name__ == "__main__":
    spy = give_regime_data()
    spy["position"] = spy["regime"].shift(1)
    spy["strategy_returns"] = spy["returns"] * spy["position"]
    spy["cumulative_strategy_returns"] = (1 + spy["strategy_returns"]).cumprod().fillna(1)
    spy["cumulative_hold"] = (1 + spy["returns"]).cumprod().fillna(1)
    print("END")
