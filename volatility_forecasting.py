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
    test_rolling20 = test[["Date","rolling_20"]].reset_index(drop=True)

    fig, ax = plt.subplots()
    ax.plot(test_rolling20["Date"], test_rolling20["rolling_20"], label='Rolling 20d vol (baseline)', color='red',
            linewidth=0.7)
    ax.plot(forecast_df['Date'], forecast_df['forecast_vol_annualized'], label='GARCH forecast vol', color='blue',
            linewidth=0.7)
    ax.legend()
    fig.savefig("figures/garch_vs_rolling20_forecast.png")

    check = forecast_df.merge(test_rolling20, on='Date').iloc[280:295]  # adjust slice to land near the spike
    print(check[['Date', 'forecast_vol_annualized', 'rolling_20']])

    print("END")


