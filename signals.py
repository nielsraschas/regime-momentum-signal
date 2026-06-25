import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




if __name__ == "__main__":
    spy = pd.read_csv("data/spy.csv", parse_dates=["Date"])
    spy["returns"] = spy["Close"].pct_change()
    spy["rolling_20"] = spy["returns"].rolling(20).std() * (252)**(1/2)
    spy["rolling_60"] = spy["returns"].rolling(60).std() * (252) ** (1 / 2)
    spy["regime"] = -1
    spy.loc[spy["rolling_20"]<=spy["rolling_60"], "regime"] = +1
    fig, ax = plt.subplots()
    ax.plot(spy["Date"], spy["rolling_20"], label="rolling_window_20")
    ax.plot(spy["Date"], spy["rolling_60"], label="rolling_window_60")
    ax.legend()
    fig.savefig("figures/rolling_60_vs_20")
    fig2, ax2 = plt.subplots()
    ax2.plot(spy["Date"], spy["returns"], color="black", linewidth=0.5)
    ax2.fill_between(spy["Date"], spy["returns"].min(), spy["returns"].max(),
                     where=spy["regime"] == 1, color="green", alpha=0.2)
    ax2.fill_between(spy["Date"], spy["returns"].min(), spy["returns"].max(),
                     where=spy["regime"] == -1, color="red", alpha=0.2)
    fig2.savefig("figures/color_plot_returns.png")
    fig3, ax3 = plt.subplots()
    ax3.plot(spy["Date"], spy["Close"], color="black", linewidth=0.5)
    ax3.fill_between(spy["Date"], spy["Close"].min(), spy["Close"].max(),
                     where=spy["regime"] == 1, color="green", alpha=0.2)
    ax3.fill_between(spy["Date"], spy["Close"].min(), spy["Close"].max(),
                     where=spy["regime"] == -1, color="red", alpha=0.2)
    fig3.savefig("figures/color_plot_prices.png")
    print("END")