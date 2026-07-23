import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import het_arch

if __name__ == "__main__":
    spy = pd.read_csv("data/spy.csv", parse_dates=["Date"])
    spy["returns"] = spy["Close"].pct_change()
    spy = spy.dropna(subset=["returns"])
    spy["volatility_rolling_20"] = spy["returns"].rolling(20).std() * (252 ** 0.5)

    fig_regime, axes = plt.subplots(2, 1)
    axes[0].plot(spy["Date"], spy["returns"])
    axes[1].plot(spy["Date"], spy["volatility_rolling_20"])
    fig_regime.savefig("figures/regime_plot.png")

    largest_vol = spy.nlargest(100, "volatility_rolling_20")[["Date", "volatility_rolling_20"]]
    print(largest_vol.sort_values("Date"))

    returns = spy["returns"].to_numpy()

    model = AutoReg(returns, lags=1).fit()
    print(model.summary())
    residuals = model.resid  # plain ndarray now, no index games

    fig_acf_resid = plot_acf(residuals, lags=20).figure
    fig_acf_resid.savefig("figures/acf_returns_residuals.png")

    for p in range(1, 6):
        m = AutoReg(returns, lags=p, old_names=False).fit()
        print(f"AR({p}): AIC={m.aic:.2f}, BIC={m.bic:.2f}")

    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(residuals)
    print(f"LM p-value: {lm_pvalue}")

    sq_resid = residuals ** 2
    fig_acf_sq = plot_acf(sq_resid, lags=20).figure
    fig_acf_sq.savefig("figures/acf_squared_residuals.png")

    print("END")