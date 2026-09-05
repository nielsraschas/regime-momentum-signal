import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from backtest import sharpe_ratio, max_drawdown

# -----------------------
# Metrics helpers
# -----------------------

def qlike(realized, forecast):
    """QLIKE loss: robust to noisy realized-variance proxies. Lower = better."""
    mask = realized > 0  # guards against log(0) when realized variance is exactly 0
    ratio = realized[mask] / forecast[mask]
    return (ratio - np.log(ratio) - 1).mean()


def get_metrics(data, modelized_name, realized_name):
    """RMSE + QLIKE of `{modelized_name}_var_daily` against `realized_name`."""
    data["se_" + modelized_name] = (data[modelized_name + "_var_daily"] - data[realized_name]) ** 2
    rmse = (data["se_" + modelized_name].mean()) ** 0.5
    qlike_value = qlike(data[realized_name], data[modelized_name + "_var_daily"])
    return rmse, qlike_value

def count_switches(data, col):
    return (data[col] != data[col].shift(1)).astype(int).sum()

def compute_regime_metrics(data, regime_col, irx_daily, label):
    """Position (lagged 1 day), TC-adjusted returns, Sharpe, max DD for a given
    regime column -- same position/TC logic as backtest.py's strategy_calculation."""
    position = data[regime_col].shift(1)
    strategy_returns = data["returns"] * position
    transaction_cost = pd.Series(0.0, index=data.index)
    transaction_cost[position != position.shift(1)] = 0.001
    strategy_returns_tc = strategy_returns - transaction_cost

    switches = count_switches(data, regime_col)
    sharpe = sharpe_ratio(strategy_returns_tc, irx_daily)
    dd = max_drawdown(strategy_returns_tc)

    print(f"{label}: switches={switches}, Sharpe(TC)={sharpe:.3f}, MaxDD={dd:.3f}")
    return dict(switches=switches, sharpe_tc=sharpe, max_dd=dd)

# ----------------------------
# GARCH-based regime signal
# -----------------------------

def build_garch_regimes(data):
    """Raw and 5-day-smoothed GARCH regime signals, same 1.2x threshold rule
    as the baseline. Smoothing uses a TRAILING window only (no lookahead --
    caught and fixed a negative-shift bug here on Aug 24)."""
    data["garch_vol_annual"] = (data["garch_var_daily"] ** 0.5) * (252 ** 0.5)
    data["regime_garch"] = 1
    data.loc[data["garch_vol_annual"] >= data["rolling_252"] * 1.2, "regime_garch"] = 0

    data["garch_vol_annual_roll"] = data["garch_vol_annual"].rolling(5).mean()
    data["regime_garch_roll"] = 1
    data.loc[data["garch_vol_annual_roll"] >= data["rolling_252"] * 1.2, "regime_garch_roll"] = 0

    return data





if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # Data loading + feature construction
    # -----------------------------------------------------------------------
    spy = pd.read_csv("data/spy.csv", parse_dates=["Date"])
    irx = pd.read_csv("data/irx.csv", parse_dates=["Date"])

    spy["returns"] = spy["Close"].pct_change()
    spy["realized_volatility"] = spy["returns"] ** 2
    spy["realized_volatility_fwd"] = sum(spy["realized_volatility"].shift(-k) for k in range(1, 6)) / 5
    spy["rolling_20"] = spy["returns"].rolling(20).std() * (252) ** 0.5
    spy["rolling_252"] = spy["returns"].rolling(252).std() * (252) ** 0.5

    # baseline regime, identical rule to backtest.py's give_regime_data()
    spy["regime"] = 1
    spy.loc[spy["rolling_20"] >= spy["rolling_252"] * 1.2, "regime"] = 0

    spy = spy.dropna(subset=["returns"])
    spy = spy.dropna(subset=["rolling_20"])
    spy = spy.dropna(subset=["realized_volatility_fwd"])



    # -----------------------------------------------------------------------
    # GARCH fitting: Normal vs Student-t (establishes t wins on AIC/BIC)
    # -----------------------------------------------------------------------
    returns_pct = spy["returns"] * 100

    am = arch_model(returns_pct, vol="Garch", p=1, q=1, mean="Constant")
    res = am.fit(disp="off")
    print(res.summary())

    am_t = arch_model(returns_pct, vol="Garch", p=1, q=1, mean="Constant", dist="t")
    res_t = am_t.fit(disp="off")
    print(res_t.summary())

    # -----------------------------------------------------------------------
    # OOT forecast: fit on train only, forecast 1-step-ahead across test
    # -----------------------------------------------------------------------
    train = spy[spy["Date"] < "2019-01-01"].copy()
    test = spy[spy["Date"] >= "2019-01-01"].copy()

    am_full = arch_model(spy["returns"] * 100, vol="Garch", p=1, q=1, mean="Constant", dist="t")
    res_full = am_full.fit(disp="off", first_obs=0, last_obs=len(train))
    print(res_full.summary())

    forecast = res_full.forecast(horizon=1, start=len(train), reindex=False)

    test_dates = test["Date"].reset_index(drop=True)
    forecast_variance = forecast.variance["h.1"].reset_index(drop=True)
    print(len(test_dates), len(forecast_variance))  # sanity check: lengths must match

    forecast_df = pd.DataFrame({
        "Date": test_dates,
        "forecast_variance": forecast_variance,
    })
    forecast_df["forecast_vol"] = (forecast_df["forecast_variance"] ** 0.5) / 100
    forecast_df["forecast_vol_annualized"] = forecast_df["forecast_vol"] * (252 ** 0.5)

    test_cols = test[["Date", "returns", "rolling_20", "rolling_252",
                      "realized_volatility", "realized_volatility_fwd", "regime"]].reset_index(drop=True)

    check = forecast_df.merge(test_cols, on="Date")
    # print(check[["Date", "forecast_vol_annualized", "rolling_20"]].iloc[280:295])  # eyeball a window

    # unit conversions: forecast_variance is in percent^2 (fit on returns*100)
    check["garch_var_daily"] = check["forecast_variance"] / 10000
    check["rolling_var_daily"] = (check["rolling_20"] / (252 ** 0.5)) ** 2
    check["se_garch"] = (check["garch_var_daily"] - check["realized_volatility"]) ** 2
    check["se_rolling"] = (check["rolling_var_daily"] - check["realized_volatility"]) ** 2

    # sanity check: all three should be the same order of magnitude (~1e-4 for SPY)
    print(check["garch_var_daily"].mean(), check["rolling_var_daily"].mean(), check["realized_volatility"].mean())
    print(check["garch_var_daily"].min(), check["rolling_var_daily"].min())  # confirm no zeros/negatives

    # -----------------------------------------------------------------------
    # Forecast-accuracy metrics -- single-day, calm-only, forward target
    # -----------------------------------------------------------------------

    rmse_garch, qlike_garch = get_metrics(check, "garch", "realized_volatility")
    rmse_rolling, qlike_rolling = get_metrics(check, "rolling", "realized_volatility")
    print(rmse_garch, rmse_rolling)
    print(qlike_garch, qlike_rolling)

    # -----------------------------------------------------------------------
    # GARCH-based regime signal -- raw vs 5-day-smoothed
    # -----------------------------------------------------------------------

    check = build_garch_regimes(check)

    print("Rolling-based switches (test period):", count_switches(check, "regime"))
    print("GARCH-based switches, raw (test period):", count_switches(check, "regime_garch"))
    print("GARCH-based switches, 5d-smoothed (test period):", count_switches(check, "regime_garch_roll"))

    # -----------------------------------------------------------------------
    # Without COVID metric exploration
    # -----------------------------------------------------------------------

    crisis = (check["Date"] >= "2020-02-15") & (check["Date"] <= "2020-04-30")
    calm = check[~crisis]
    rmse_garch_nocovid, qlike_garch_nocovid = get_metrics(calm, "garch", "realized_volatility")
    rmse_rolling_nocovid, qlike_rolling_nocovid = get_metrics(calm, "rolling", "realized_volatility")
    print(rmse_garch_nocovid, rmse_rolling_nocovid)
    print(qlike_garch_nocovid, qlike_rolling_nocovid)

    rmse_garch_fwd, qlike_garch_fwd = get_metrics(check, "garch", "realized_volatility_fwd")
    rmse_rolling_fwd, qlike_rolling_fwd = get_metrics(check, "rolling", "realized_volatility_fwd")
    print(rmse_garch_fwd, rmse_rolling_fwd)
    print(qlike_garch_fwd, qlike_rolling_fwd)

    # -----------------------------------------------------------------------
    # Sharpe/TC on regime_garch_roll vs regime, same test period
    # -----------------------------------------------------------------------

    irx_adjusted = pd.merge(check[["Date"]], irx, on="Date", how="left")
    irx_adjusted["Close"] = irx_adjusted["Close"].ffill().astype(float)
    irx_daily = irx_adjusted["Close"] / 100 / 252

    baseline_full = compute_regime_metrics(check, "regime", irx_daily, "regime")
    garch_full = compute_regime_metrics(check, "regime_garch_roll", irx_daily, "regime_garch_roll")

    irx_daily_calm = irx_daily[~crisis.values]

    baseline_calm = compute_regime_metrics(calm, "regime", irx_daily_calm, "regime")
    garch_calm = compute_regime_metrics(calm, "regime_garch_roll", irx_daily_calm, "regime_garch_roll")

    # -----------------------------------------------------------------------
    # GARCH tearsheet -- 3-panel companion to backtest.py's tearsheet.png
    # -----------------------------------------------------------------------
    check["cumulative_hold"] = (1 + check["returns"]).cumprod()

    for col in ["regime", "regime_garch_roll"]:
        pos = check[col].shift(1)
        strat_ret = check["returns"] * pos
        tc = pd.Series(0.0, index=check.index)
        tc[pos != pos.shift(1)] = 0.001
        check[f"cumulative_{col}_tc"] = (1 + strat_ret - tc).cumprod()

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), gridspec_kw={'height_ratios': [3, 3, 1]})

    # Panel 1: equity curves, test period, TC-adjusted
    axes[0].plot(check["Date"], check["cumulative_hold"], label="Buy & Hold", color="red")
    axes[0].plot(check["Date"], check["cumulative_regime_tc"], label="Baseline Regime (rolling_20)", color="green")
    axes[0].plot(check["Date"], check["cumulative_regime_garch_roll_tc"], label="GARCH Regime (5d-smoothed)",
                 color="blue")
    axes[0].legend()
    axes[0].set_title("Cumulative Returns, Out-of-Time Test Period (with transaction costs)")

    # Panel 2: forecast accuracy overlay -- why the strategies differ
    axes[1].plot(check["Date"], check["rolling_20"], color="red", linewidth=0.7, label="Rolling 20d vol (baseline)")
    axes[1].plot(check["Date"], check["garch_vol_annual"], color="blue", linewidth=0.7, label="GARCH forecast vol")
    axes[1].legend()
    axes[1].set_title("Forecast Accuracy: GARCH vs Rolling-Window Volatility")

    # Panel 3: text metrics table
    axes[2].axis("off")
    metrics_text = (
        f"{'Metric':<24}{'Baseline':>12}{'GARCH':>12}\n"
        f"{'Sharpe (full, TC)':<24}{baseline_full['sharpe_tc']:>12.2f}{garch_full['sharpe_tc']:>12.2f}\n"
        f"{'Sharpe (ex-COVID, TC)':<24}{baseline_calm['sharpe_tc']:>12.2f}{garch_calm['sharpe_tc']:>12.2f}\n"
        f"{'Max Drawdown':<24}{baseline_full['max_dd']:>11.1%}{garch_full['max_dd']:>12.1%}\n"
        f"{'Regime switches':<24}{baseline_full['switches']:>12d}{garch_full['switches']:>12d}\n"
        f"{'  (GARCH raw, unsmoothed: ' + str(count_switches(check, 'regime_garch')) + ' switches)':<48}\n"
    )
    axes[2].text(0.05, 0.95, metrics_text, fontsize=10, family="monospace",
                 verticalalignment="top")

    fig.tight_layout()
    fig.savefig("figures/garch_tearsheet.png")

    print("END")


