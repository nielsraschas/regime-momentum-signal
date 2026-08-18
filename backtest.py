import pandas as pd
import matplotlib.pyplot as plt


def give_regime_data():
    spy = pd.read_csv("data/spy.csv", parse_dates=["Date"])
    spy = spy.sort_values("Date").reset_index(drop=True)
    spy["returns"] = spy["Close"].pct_change()
    spy["rolling_20"] = spy["returns"].rolling(20).std() * (252) ** (1 / 2)
    spy["rolling_252"] = spy["returns"].rolling(252).std() * (252) ** (1 / 2)
    spy["regime"] = 1
    spy.loc[spy["rolling_20"] >= spy["rolling_252"] * 1.2, "regime"] = 0
    return spy


def sharpe_ratio(data_return, irx_close):
    data_adjusted_return = data_return - irx_close
    return (data_adjusted_return.mean() / data_adjusted_return.std()) * (252) ** (1 / 2)


def max_drawdown(returns):
    wealth_index = (1 + returns).cumprod().fillna(1)
    drawdown = (wealth_index.cummax() - wealth_index) / wealth_index.cummax()
    return max(drawdown)


def strategy_calculation(spy):
    spy["cumulative_hold"] = (1 + spy["returns"]).cumprod().fillna(1)

    ##### Regime change integrated
    spy["position"] = spy["regime"].shift(1)
    spy["strategy_returns"] = spy["returns"] * spy["position"]
    spy["cumulative_strategy_returns"] = (1 + spy["strategy_returns"]).cumprod().fillna(1)
    spy["regime_shift"] = 0
    spy.loc[spy["regime"] != spy["regime"].shift(1), "regime_shift"] = 1
    spy["cumulative_regime_shift"] = spy["regime_shift"].cumsum()

    ##### Momentum integrated
    spy["momentum"] = spy["returns"].rolling(20).sum()
    spy["position_combined"] = 0
    spy.loc[(spy["regime"].shift(1) == 1) & (spy["momentum"].shift(1) > 0), "position_combined"] = 1
    spy["strategy_combined_returns"] = spy["returns"] * spy["position_combined"]
    spy["cumulative_strategy_combined_returns"] = (1 + spy["strategy_combined_returns"]).cumprod().fillna(1)
    spy["combined_shift"] = 0
    spy.loc[spy["position_combined"] != spy["position_combined"].shift(1), "combined_shift"] = 1

    ##### Transaction cost integrated
    spy["transaction_cost"] = 0.00
    spy.loc[(spy["position"] != spy["position"].shift(1)), "transaction_cost"] = 0.001  # 0.1% cost
    spy["transaction_cost_combined"] = 0.00
    spy.loc[(spy["position_combined"] != spy["position_combined"].shift(
        1)), "transaction_cost_combined"] = 0.001  # 0.1% cost
    spy["cumulative_strategy_returns_tc"] = (1 + spy["strategy_returns"] - spy["transaction_cost"]).cumprod().fillna(1)
    spy["cumulative_strategy_combined_returns_tc"] = (
            1 + spy["strategy_combined_returns"] - spy["transaction_cost_combined"]).cumprod().fillna(1)

    return spy


def metrics_and_graphs(spy, irx_daily, label_strat=""):
    spy = strategy_calculation(spy)
    sharp_ratio_hold = sharpe_ratio(spy["returns"], irx_daily)
    max_drawdown_hold = max_drawdown(spy["returns"])
    sharp_ratio_strategy_tc = sharpe_ratio(spy["strategy_returns"] - spy["transaction_cost"], irx_daily)
    sharp_ratio_combined_strategy_tc = sharpe_ratio(spy["strategy_combined_returns"] - spy["transaction_cost_combined"],
                                                    irx_daily)
    max_drawdown_strategy_tc = max_drawdown(spy["strategy_returns"] - spy["transaction_cost"])
    max_drawdown_combined_strategy_tc = max_drawdown(
        spy["strategy_combined_returns"] - spy["transaction_cost_combined"])
    fig, ax = plt.subplots()
    ax.plot(spy["Date"], spy["cumulative_hold"], color="red", linewidth=0.5, label="cumulative_hold" + label_strat)
    ax.plot(spy["Date"], spy["cumulative_strategy_returns_tc"], color="green", linewidth=0.5,
            label="cumulative_strategy_returns_tc" + label_strat)
    ax.plot(spy["Date"], spy["cumulative_strategy_combined_returns_tc"], color="blue", linewidth=0.5,
            label="cumulative_combined_strategy_returns_tc" + label_strat)
    ax.set_title(
        f"Cumulative switches: {spy['cumulative_regime_shift'].max()}\n"
        f"Buy & Hold: Sharpe={str(round(sharp_ratio_hold, 2))} | Max DD={str(round(max_drawdown_hold, 3))}\n"
        f"Strategy: Sharpe={str(round(sharp_ratio_strategy_tc, 2))} | Max DD={str(round(max_drawdown_strategy_tc, 3))}\n"
        f"Combined Strategy: Sharpe={str(round(sharp_ratio_combined_strategy_tc, 2))} | Max DD={str(round(max_drawdown_combined_strategy_tc, 3))}",
        fontsize=9)
    fig.savefig("figures/hold_vs_strategy_vs_combined_tc" + label_strat + ".png")
    print(spy["cumulative_hold"].iloc[-1])
    print(spy["cumulative_strategy_returns_tc"].iloc[-1])
    print(spy["cumulative_strategy_combined_returns_tc"].iloc[-1])
    print(spy["combined_shift"].cumsum().max())
    return sharp_ratio_hold, max_drawdown_hold, sharp_ratio_strategy_tc, max_drawdown_strategy_tc, sharp_ratio_combined_strategy_tc, max_drawdown_combined_strategy_tc

def compute_metrics(spy, irx_daily):
    sharpe_hold = sharpe_ratio(spy["returns"], irx_daily)
    dd_hold = max_drawdown(spy["returns"])
    sharpe_strategy = sharpe_ratio(spy["strategy_returns"] - spy["transaction_cost"], irx_daily)
    sharpe_combined = sharpe_ratio(spy["strategy_combined_returns"] - spy["transaction_cost_combined"],
                                                    irx_daily)
    dd_strategy = max_drawdown(spy["strategy_returns"] - spy["transaction_cost"])
    dd_combined = max_drawdown(
        spy["strategy_combined_returns"] - spy["transaction_cost_combined"])
    return dict(sharpe_hold=sharpe_hold, sharpe_strategy=sharpe_strategy, sharpe_combined=sharpe_combined,
                dd_hold=dd_hold, dd_strategy=dd_strategy, dd_combined=dd_combined)


def walk_forward_evaluate(spy, irx_daily, n_splits=5):
    spy = strategy_calculation(spy)
    fold_bounds = pd.qcut(spy.index, n_splits, labels=False)
    results = []
    for fold in range(n_splits):
        mask = fold_bounds == fold
        results.append(compute_metrics(spy[mask], irx_daily[mask]))
    return pd.DataFrame(results)


if __name__ == "__main__":
    spy = give_regime_data()
    irx = pd.read_csv("data/irx.csv", parse_dates=["Date"])
    irx_adjusted = pd.merge(spy[["Date"]], irx, on="Date", how="left")
    irx_adjusted["Close"] = irx_adjusted["Close"].ffill().apply(lambda x: float(x))
    irx_daily = irx_adjusted["Close"] / 100 / 252
    sharp_ratio_hold, max_drawdown_hold, sharp_ratio_strategy_tc, max_drawdown_strategy_tc, sharp_ratio_combined_strategy_tc, max_drawdown_combined_strategy_tc = metrics_and_graphs(
        spy, irx_daily, label_strat="")
    train_spy = spy[spy["Date"] < "2019-01-01"].copy()
    test_spy = spy[spy["Date"] >= "2019-01-01"].copy()
    sharp_ratio_hold_train, max_drawdown_hold_train, sharp_ratio_strategy_tc_train, max_drawdown_strategy_tc_train, sharp_ratio_combined_strategy_tc_train, max_drawdown_combined_strategy_tc_train = metrics_and_graphs(
        train_spy, irx_daily, label_strat="_train")
    sharp_ratio_hold_test, max_drawdown_hold_test, sharp_ratio_strategy_tc_test, max_drawdown_strategy_tc_test, sharp_ratio_combined_strategy_tc_test, max_drawdown_combined_strategy_tc_test = metrics_and_graphs(
        test_spy, irx_daily, label_strat="_test")

    wf_df = walk_forward_evaluate(spy, irx_daily, n_splits=5)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), gridspec_kw={'height_ratios': [3, 3, 1]})

    # Panel 1: equity curves (all three strategies)
    axes[0].plot(spy["Date"], spy["cumulative_hold"], label="Buy & Hold", color="red")
    axes[0].plot(spy["Date"], spy["cumulative_strategy_returns_tc"], label="Regime", color="green")
    axes[0].plot(spy["Date"], spy["cumulative_strategy_combined_returns_tc"], label="Combined", color="blue")
    axes[0].legend()
    axes[0].set_title("Cumulative Returns (with transaction costs)")

    # Panel 2: regime overlay on price
    axes[1].plot(spy["Date"], spy["Close"], color="black", linewidth=0.5)
    axes[1].fill_between(spy["Date"], spy["Close"].min(), spy["Close"].max(),
                         where=spy["regime"] == 1, color="green", alpha=0.25)
    axes[1].fill_between(spy["Date"], spy["Close"].min(), spy["Close"].max(),
                         where=spy["regime"] == 0, color="red", alpha=0.25)
    axes[1].set_title("SPY Price with Regime Overlay (green=low-vol, red=high-vol)")

    # Panel 3: could be a text-based metrics summary, or IS/OOS bar comparison
    axes[2].axis("off")
    metrics_text = (
        f"{'Metric':<20}{'Buy&Hold':>10}{'Regime':>10}{'Combined':>10}\n"
        f"{'Sharpe (full)':<20}{sharp_ratio_hold:>10.2f}{sharp_ratio_strategy_tc:>10.2f}{sharp_ratio_combined_strategy_tc:>10.2f}\n"
        f"{'Max DD (full)':<20}{max_drawdown_hold:>10.1%}{max_drawdown_strategy_tc:>10.1%}{max_drawdown_combined_strategy_tc:>10.1%}\n"
        f"{'Sharpe (train)':<20}{sharp_ratio_hold_train:>10.2f}{sharp_ratio_strategy_tc_train:>10.2f}{sharp_ratio_combined_strategy_tc_train:>10.2f}\n"
        f"{'Max DD (train)':<20}{max_drawdown_hold_train:>10.1%}{max_drawdown_strategy_tc_train:>10.1%}{max_drawdown_combined_strategy_tc_train:>10.1%}\n"
        f"{'Sharpe (test)':<20}{sharp_ratio_hold_test:>10.2f}{sharp_ratio_strategy_tc_test:>10.2f}{sharp_ratio_combined_strategy_tc_test:>10.2f}\n"
        f"{'Max DD (test)':<20}{max_drawdown_hold_test:>10.1%}{max_drawdown_strategy_tc_test:>10.1%}{max_drawdown_combined_strategy_tc_test:>10.1%}\n"
    )
    axes[2].text(0.05, 0.95, metrics_text, fontsize=10, family="monospace",
                 verticalalignment="top")

    fig.tight_layout()
    fig.savefig("figures/tearsheet.png")
    ###

    print("END")
