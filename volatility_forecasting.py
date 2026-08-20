import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model


def qlike(realized, forecast):
    mask = realized > 0
    ratio = realized[mask] / forecast[mask]
    return (ratio - np.log(ratio) - 1).mean()

def get_metrics(data, modelized_name, realized_name):
    data["se_" + modelized_name] = (data[modelized_name + "_var_daily"] - data[realized_name]) ** 2
    rmse = (data["se_" + modelized_name].sum() / len(data["se_" + modelized_name])) ** 0.5
    qlike_value = qlike(data[realized_name], data[modelized_name + "_var_daily"])
    return rmse, qlike_value


if __name__ == "__main__":
    spy = pd.read_csv("data/spy.csv", parse_dates=["Date"])
    spy["returns"] = spy["Close"].pct_change()
    spy["realized_volatility"] = spy["returns"]**2
    spy["realized_volatility_fwd"] = sum(spy["realized_volatility"].shift(-k) for k in range(1, 6)) / 5
    spy["rolling_20"] = spy["returns"].rolling(20).std()*(252)**(1/2)
    spy = spy.dropna(subset=["returns"])
    spy = spy.dropna(subset=["rolling_20"])
    spy = spy.dropna(subset=["realized_volatility_fwd"])

    returns_pct = spy["returns"] * 100

    am = arch_model(returns_pct, vol="Garch", p=1, q=1, mean="Constant")
    res = am.fit(disp="off")
    print(res.summary())

    am_t = arch_model(returns_pct, vol="Garch", p=1, q=1, mean='Constant', dist='t')
    res_t = am_t.fit(disp="off")
    print(res_t.summary())

    train = spy[spy['Date']<"2019-01-01"].copy()
    test = spy[spy['Date'] >= "2019-01-01"].copy()

    am_full = arch_model(spy['returns']*100, vol="Garch", p=1, q=1, mean='Constant', dist='t')
    res_full = am_full.fit(disp="off", first_obs=0, last_obs=len(train))
    print(res_full.summary())

    forecast = res_full.forecast(horizon=1, start=len(train), reindex=False)

    test_dates = test['Date'].reset_index(drop=True)
    forecast_variance = forecast.variance['h.1'].reset_index(drop=True)

    print(len(test_dates), len(forecast_variance))

    forecast_df = pd.DataFrame({
        'Date': test_dates,
        'forecast_variance': forecast_variance
    })

    forecast_df['forecast_vol'] = (forecast_df['forecast_variance'] ** 0.5) / 100
    forecast_df['forecast_vol_annualized'] = forecast_df['forecast_vol'] * (252 ** 0.5)
    test_rolling20 = test[["Date","rolling_20", "realized_volatility", "realized_volatility_fwd"]].reset_index(drop=True)

    fig, ax = plt.subplots()
    ax.plot(test_rolling20["Date"], test_rolling20["rolling_20"], label='Rolling 20d vol (baseline)', color='red',
            linewidth=0.7)
    ax.plot(forecast_df['Date'], forecast_df['forecast_vol_annualized'], label='GARCH forecast vol', color='blue',
            linewidth=0.7)
    ax.legend()
    fig.savefig("figures/garch_vs_rolling20_forecast.png")



    check = forecast_df.merge(test_rolling20, on='Date')
    print(check[['Date', 'forecast_vol_annualized', 'rolling_20']].iloc[280:295])

    check["garch_var_daily"] = check["forecast_variance"] / 10000
    check["rolling_var_daily"] = (check["rolling_20"] / (252 ** 0.5)) ** 2

    check["se_garch"] = (check["garch_var_daily"] - check["realized_volatility"])**2
    rmse_garch = (check["se_garch"].sum()/len(check["se_garch"]))**0.5

    check["se_rolling"] = (check["rolling_var_daily"] - check["realized_volatility"]) ** 2
    rmse_rolling = (check["se_rolling"].sum() / len(check["se_rolling"]))**0.5

    print(check["garch_var_daily"].mean(), check["rolling_var_daily"].mean(), check["realized_volatility"].mean())
    print(rmse_garch, rmse_rolling)
    print(check["garch_var_daily"].min(), check["rolling_var_daily"].min())

    qlike_garch = qlike(check["realized_volatility"], check["garch_var_daily"])
    qlike_rolling = qlike(check["realized_volatility"], check["rolling_var_daily"])
    print(qlike_garch, qlike_rolling)

    crisis = (check["Date"] >= "2020-02-15") & (check["Date"] <= "2020-04-30")
    calm = check[~crisis]
    rmse_garch_nocovid = (calm["se_garch"].sum() / len(calm["se_garch"])) ** 0.5
    rmse_rolling_nocovid = (calm["se_rolling"].sum() / len(calm["se_rolling"])) ** 0.5
    print(rmse_garch_nocovid, rmse_rolling_nocovid)
    qlike_garch_nocovid = qlike(calm["realized_volatility"], calm["garch_var_daily"])
    qlike_rolling_nocovid = qlike(calm["realized_volatility"], calm["rolling_var_daily"])
    print(qlike_garch_nocovid, qlike_rolling_nocovid)

    rmse_garch_fwd, qlike_garch_fwd = get_metrics(check, "garch", "realized_volatility_fwd")
    rmse_rolling_fwd, qlike_rolling_fwd = get_metrics(check, "rolling", "realized_volatility_fwd")
    print(rmse_garch_fwd, rmse_rolling_fwd)
    print(qlike_garch_fwd, qlike_rolling_fwd)

    print("END")


