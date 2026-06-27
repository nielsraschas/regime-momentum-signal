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

def sharp_ratio(data_return, irx_close):
    data_adjusted_return = data_return - irx_close
    return  (data_adjusted_return.mean()/ data_adjusted_return.std()) * (252) ** (1/2)

def max_drawdown(returns):
    wealth_index = (1 + returns).cumprod().fillna(1)
    drawdown = (wealth_index.cummax() - wealth_index)/wealth_index.cummax()
    return max(drawdown)




if __name__ == "__main__":
    spy = give_regime_data()
    irx = pd.read_csv("data/irx.csv", parse_dates=["Date"])
    irx_adjusted = pd.merge(spy[["Date"]], irx, on="Date", how="left")
    irx_adjusted["Close"] = irx_adjusted["Close"].ffill().apply(lambda x: float(x))
    irx_daily = irx_adjusted["Close"] / 100 / 252
    spy["position"] = spy["regime"].shift(1)
    spy["strategy_returns"] = spy["returns"] * spy["position"]
    spy["cumulative_strategy_returns"] = (1 + spy["strategy_returns"]).cumprod().fillna(1)
    spy["cumulative_hold"] = (1 + spy["returns"]).cumprod().fillna(1)
    spy["regime_shift"] = 0
    spy.loc[spy["regime"]!= spy["regime"].shift(1), "regime_shift"] = 1
    spy["cumulative_regime_shift"] = spy["regime_shift"] .cumsum()
    sharp_ratio_strategy = sharp_ratio(spy["strategy_returns"], irx_daily)
    sharp_ratio_hold = sharp_ratio(spy["returns"], irx_daily)
    max_drawdown_strategy = max_drawdown(spy["strategy_returns"])
    max_drawdown_hold = max_drawdown(spy["returns"])
    fig, ax = plt.subplots()
    ax.plot(spy["Date"],spy["cumulative_hold"], color="red", linewidth=0.5, label="cumulative_hold")
    ax.plot(spy["Date"], spy["cumulative_strategy_returns"], color="green", linewidth=0.5, label="cumulative_strategy_returns")
    ax.set_title(
        f"Cumulative switches: {spy['cumulative_regime_shift'].max()}\n"
        f"Buy & Hold: Sharpe={str(round(sharp_ratio_hold, 2))} | Max DD={str(round(max_drawdown_hold, 3))}\n"
        f"Strategy: Sharpe={str(round(sharp_ratio_strategy, 2))} | Max DD={str(round(max_drawdown_strategy, 3))}",
        fontsize=9)
    fig.savefig("figures/hold_vs_strategy.png")
    print(spy["cumulative_hold"].iloc[-1])
    print(spy["cumulative_strategy_returns"].iloc[-1])
    print("END")
