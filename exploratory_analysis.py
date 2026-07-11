import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    spy = pd.read_csv("data/spy.csv", parse_dates=["Date"])
    spy["returns"] = spy["Close"].pct_change()
    spy["volatility_rolling_20"] = spy["returns"].rolling(20).std()*(252)**(1/2)
    figure, axes = plt.subplots(2,1)
    axes[0].plot(spy["Date"], spy["returns"])
    axes[1].plot(spy["Date"], spy["volatility_rolling_20"])
    # figure.show()
    figure.savefig("figures/regime_plot.png")
    largest_vol = spy.nlargest(100, "volatility_rolling_20")[["Date", "volatility_rolling_20"]]
    print(largest_vol.sort_values("Date"))
    print("END")