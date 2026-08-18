import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model


if __name__ == "__main__":
    spy = pd.read_csv("data/spy.csv", parse_dates=["Date"])
    spy["returns"] = spy["Close"].pct_change()
    spy["rolling_20"] = spy["returns"].rolling(20).std()*(252)**(1/2)
    spy = spy.dropna(subset=["returns"])
    spy = spy.dropna(subset=["rolling_20"])

    returns_pct = spy["returns"] * 100

    am = arch_model(returns_pct, vol="Garch", p=1, q=1, mean="Constant")
    res = am.fit(disp="off")
    print(res.summary())
    print("END")


