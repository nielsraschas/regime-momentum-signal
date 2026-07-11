import pandas as pd
import matplotlib.pyplot as plt


"""
Exploratory script used to test regime-detection parameter robustness 
(threshold 1.1-1.4, window pairs 20d/60d vs 20d/252d) before selecting 
the final specification (20d/252d, threshold=1.2) used in backtest.py.
Generates the switch-count figures referenced in the README methodology.
"""



if __name__ == "__main__":
    spy = pd.read_csv("data/spy.csv", parse_dates=["Date"])
    spy["returns"] = spy["Close"].pct_change()
    spy["rolling_20"] = spy["returns"].rolling(20).std() * (252) ** (1/2)
    spy["rolling_60"] = spy["returns"].rolling(60).std() * (252) ** (1 / 2)
    spy["rolling_252"] = spy["returns"].rolling(252).std() * (252) ** (1 / 2)
    spy["momentum"] = spy["returns"].rolling(20).sum()
    for i in [1.1, 1.2, 1.3, 1.4]:
        spy["regime_" + str(i)] = +1
        spy.loc[spy["rolling_20"]>=spy["rolling_60"]*i, "regime_" + str(i)] = -1
        spy["regime_2_" + str(i)] = +1
        spy.loc[spy["rolling_20"] >= spy["rolling_252"] * i, "regime_2_" + str(i)] = -1
        spy["switch" + "_" + str(i)] = 0
        spy.loc[ spy["regime_" + str(i)] != spy["regime_" + str(i)].shift(1),"switch" + "_" + str(i)] = 1
        spy["switch_count" + "_" + str(i)] = spy["switch" + "_" + str(i)].cumsum()
        spy["switch_count_rolling_20" + "_" + str(i)] = spy["switch" + "_" + str(i)].rolling(20).sum().fillna(0)
        spy["switch_count_rolling_60" + "_" + str(i)] = spy["switch" + "_" + str(i)].rolling(60).sum().fillna(0)
        fig4, ax4 = plt.subplots(3, 1)
        ax4[0].plot(spy["Date"], spy["switch_count" + "_" + str(i)], color="black", linewidth=0.5)
        ax4[0].set_title(f"Cumulative switches: {spy['switch_count_' + str(i)].max()}", fontsize=9)
        ax4[1].plot(spy["Date"], spy["switch_count_rolling_20" + "_" + str(i)], color="black", linewidth=0.5)
        ax4[2].plot(spy["Date"], spy["switch_count_rolling_60" + "_" + str(i)], color="black", linewidth=0.5)
        fig4.savefig("figures/switch_count_rollings_20_60_vs_time" + "_" + str(i)+".png")
        print(spy["switch_count" + "_" + str(i)].max())
        spy["switch_2" + "_" + str(i)] = 0
        spy.loc[spy["regime_2_" + str(i)] != spy["regime_2_" + str(i)].shift(1), "switch_2" + "_" + str(i)] = 1
        spy["switch_count_2" + "_" + str(i)] = spy["switch_2" + "_" + str(i)].cumsum()
        spy["switch_count_rolling_20_2" + "_" + str(i)] = spy["switch_2" + "_" + str(i)].rolling(20).sum().fillna(0)
        spy["switch_count_rolling_252_2" + "_" + str(i)] = spy["switch_2" + "_" + str(i)].rolling(252).sum().fillna(0)
        fig5, ax5 = plt.subplots(3, 1)
        ax5[0].plot(spy["Date"], spy["switch_count_2" + "_" + str(i)], color="black", linewidth=0.5)
        ax5[0].set_title(f"Cumulative switches: {spy['switch_count_2_' + str(i)].max()}", fontsize=9)
        ax5[1].plot(spy["Date"], spy["switch_count_rolling_20_2" + "_" + str(i)], color="black", linewidth=0.5)
        ax5[2].plot(spy["Date"], spy["switch_count_rolling_252_2" + "_" + str(i)], color="black", linewidth=0.5)
        fig5.savefig("figures/switch_count_rollings_20_252_vs_time" + "_" + str(i) + ".png")
        print(spy["switch_count_2" + "_" + str(i)].max())
    print("END")