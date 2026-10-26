"""Paris winter HDD options: temperature model, pricing and out-of-sample backtest.

    python run.py              # downloads Paris daily temperatures (Open-Meteo, ERA5) once, then caches
    python run.py --synthetic  # offline run on synthetic data with known parameters
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import load_or_download, synthetic_paris
from src.model import TemperatureModel, day_of_year_365
from src.pricing import (HDDContract, backtest, burn_analysis, mc_price, realised_hdd,
                         simulate_season)

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
N_PATHS = 20_000
TICK, CAP = 20.0, None
BACKTEST_CUTOFF = 2010


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"figures/{name}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(synthetic: bool):
    Path("figures").mkdir(exist_ok=True)
    df = synthetic_paris() if synthetic else load_or_download("data/paris_daily.csv")
    last = df["date"].iloc[-1]
    model = TemperatureModel.fit(df["date"], df["temp"])
    diag = model.diagnostics

    # the next winter not yet started
    season = last.year if last < pd.Timestamp(f"{last.year}-11-01") else last.year + 1
    sims = simulate_season(model, df, season, N_PATHS, seed=7)
    past = list(range(int(df["date"].dt.year.min()), season))
    burn = burn_analysis(df, model, season, past)
    strike = float(np.round(np.median(sims), -1))
    maturity = ((pd.Timestamp(f"{season + 1}-03-31") - last).days) / 365.25
    call = HDDContract(season, strike, tick=TICK, cap=CAP)
    price, se = mc_price(call, sims, maturity_years=maturity)
    price_rp, _ = mc_price(call, simulate_season(model, df, season, N_PATHS, seed=7,
                                                 market_price_of_risk=0.05), maturity_years=maturity)
    burn_raw_price = float(call.payoff(burn["hdd_raw"]).mean() * np.exp(-0.03 * maturity))
    burn_det_price = float(call.payoff(burn["hdd_detrended"]).mean() * np.exp(-0.03 * maturity))

    bt, _ = backtest(df, BACKTEST_CUTOFF, list(range(BACKTEST_CUTOFF, season)), n_paths=5000)
    rmse = lambda c: float(np.sqrt(np.mean((bt[c] - bt["realised"]) ** 2)))

    results = {
        "data": {"source": "synthetic" if synthetic else "Open-Meteo archive (ERA5), Paris",
                 "first_day": str(df["date"].iloc[0].date()), "last_day": str(last.date()), "days": len(df)},
        "model": {
            "warming_trend_C_per_decade": round(model.trend_per_decade, 3),
            "ar_order_bic": diag["ar_order"],
            "ar_coefficients": [round(x, 4) for x in model.phi],
            "daily_vol_winter_C": round(float(np.sqrt(model.seasonal_variance(pd.DatetimeIndex(["2001-01-15"]))[0])), 2),
            "daily_vol_summer_C": round(float(np.sqrt(model.seasonal_variance(pd.DatetimeIndex(["2001-07-15"]))[0])), 2),
            "residual_acf_lag1_before": round(float(diag["resid_acf"][1]), 3),
            "residual_acf_lag1_after": round(float(diag["z_acf"][1]), 3),
            "z_skew": round(diag["z_skew"], 3),
            "z_excess_kurtosis": round(diag["z_excess_kurtosis"], 3),
        },
        "pricing": {
            "season": f"{season}-{str(season + 1)[-2:]}",
            "hdd_mean": round(float(sims.mean()), 1), "hdd_std": round(float(sims.std()), 1),
            "hdd_q05": round(float(np.quantile(sims, 0.05)), 1), "hdd_q95": round(float(np.quantile(sims, 0.95)), 1),
            "strike": strike, "tick_eur": TICK,
            "call_price_eur": round(price, 0), "call_price_se_eur": round(se, 0),
            "call_price_lambda_0.05_eur": round(price_rp, 0),
            "burn_raw_price_eur": round(burn_raw_price, 0),
            "burn_detrended_price_eur": round(burn_det_price, 0),
            "burn_raw_hdd_mean": round(float(burn["hdd_raw"].mean()), 1),
            "burn_detrended_hdd_mean": round(float(burn["hdd_detrended"].mean()), 1),
        },
        "backtest": {
            "fit_until": f"{BACKTEST_CUTOFF}-10-31", "seasons": len(bt),
            "coverage_90": round(float(((bt["realised"] > bt["model_q05"]) & (bt["realised"] < bt["model_q95"])).mean()), 3),
            "mean_pit": round(float(bt["pit"].mean()), 3),
            "rmse_model": round(rmse("model_mean"), 1),
            "rmse_burn_raw": round(rmse("burn_raw"), 1),
            "rmse_burn_detrended": round(rmse("burn_detrended"), 1),
            "table": bt.round(1).to_dict(orient="records"),
        },
    }
    Path("results.json").write_text(json.dumps(results, indent=2))

    # ---------------------------------------------------------------- figures
    # 1. data and seasonal mean (last 4 years)
    recent = df[df["date"] >= last - pd.Timedelta(days=4 * 365)]
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=SURFACE)
    ax.plot(recent["date"], recent["temp"], color=BLUE, linewidth=0.8, alpha=0.7, label="Daily (Tmax + Tmin) / 2")
    ax.plot(recent["date"], model.seasonal_mean(recent["date"]), color=ORANGE, linewidth=2, label="Trend + seasonal mean")
    ax.axhline(18, color=INK2, linewidth=1, linestyle=":")
    ax.annotate("HDD base 18 °C", (recent["date"].iloc[5], 18.4), color=INK2, fontsize=9)
    ax.legend(frameon=False, labelcolor=INK, loc="lower left")
    style(ax, "Paris daily temperature and fitted seasonal mean", "", "°C")
    save(fig, "temperature_fit")

    # 2. seasonal volatility
    dates = df["date"].iloc[len(df) - len(diag["z"]):]
    resid_e = diag["z"] * np.sqrt(model.seasonal_variance(dates))
    emp = pd.Series(resid_e ** 2).groupby(day_of_year_365(dates)).mean()
    grid_dates = pd.date_range("2001-01-01", "2001-12-31")
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=SURFACE)
    ax.plot(emp.index, np.sqrt(emp.values), color=BLUE, linewidth=1, alpha=0.6, label="Empirical, by day of year")
    ax.plot(np.arange(365), np.sqrt(model.seasonal_variance(grid_dates)), color=ORANGE, linewidth=2, label="Fitted σ(d)")
    ax.set_xticks([0, 59, 120, 181, 243, 304], ["Jan", "Mar", "May", "Jul", "Sep", "Nov"])
    ax.legend(frameon=False, labelcolor=INK)
    style(ax, "Daily temperature shocks are larger in winter", "", "Std of AR residual (°C)")
    save(fig, "seasonal_volatility")

    # 3. residual diagnostics
    lags = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=SURFACE)
    ax.plot(lags, diag["resid_acf"][1:], color=ORANGE, marker="o", linewidth=2, markersize=6, label="Deseasonalised temperature")
    ax.plot(lags, diag["e2_acf"][1:], color=YELLOW, marker="o", linewidth=2, markersize=6, label="Squared AR residuals")
    ax.plot(lags, diag["z_acf"][1:], color=BLUE, marker="o", linewidth=2, markersize=6, label="Standardised residuals")
    ax.plot(lags, diag["z2_acf"][1:], color=AQUA, marker="o", linewidth=2, markersize=6, label="Squared standardised residuals")
    band = 1.96 / np.sqrt(len(diag["z"]))
    ax.axhspan(-band, band, color=GRID, alpha=0.8)
    ax.legend(frameon=False, labelcolor=INK)
    style(ax, "Autocorrelation removed by the AR + seasonal volatility model", "Lag (days)", "Autocorrelation")
    save(fig, "residual_diagnostics")

    # 4. HDD distribution vs burn
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=SURFACE)
    ax.hist(sims, bins=60, color=BLUE, alpha=0.85, density=True, label=f"Model, {N_PATHS:,} paths")
    for col, c, lab in (("hdd_raw", ORANGE, "Past winters, raw"), ("hdd_detrended", AQUA, "Past winters, detrended")):
        ax.axvline(burn[col].mean(), color=c, linewidth=2, label=f"{lab}: mean {burn[col].mean():.0f}")
    ax.axvline(sims.mean(), color=INK, linewidth=2, linestyle="--", label=f"Model mean {sims.mean():.0f}")
    ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    style(ax, f"Paris HDD, winter {season}-{str(season + 1)[-2:]} (Nov–Mar, base 18 °C)", "HDD", "Density")
    save(fig, "hdd_distribution")

    # 5. call price vs strike
    strikes = np.linspace(np.quantile(sims, 0.05), np.quantile(sims, 0.95), 25)
    model_p = [mc_price(HDDContract(season, k, TICK), sims, maturity_years=maturity)[0] for k in strikes]
    raw_p = [HDDContract(season, k, TICK).payoff(burn["hdd_raw"]).mean() * np.exp(-0.03 * maturity) for k in strikes]
    det_p = [HDDContract(season, k, TICK).payoff(burn["hdd_detrended"]).mean() * np.exp(-0.03 * maturity) for k in strikes]
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=SURFACE)
    ax.plot(strikes, model_p, color=BLUE, linewidth=2, label="Model (Monte Carlo)")
    ax.plot(strikes, raw_p, color=ORANGE, linewidth=2, label="Burn analysis, raw")
    ax.plot(strikes, det_p, color=AQUA, linewidth=2, label="Burn analysis, detrended")
    ax.legend(frameon=False, labelcolor=INK)
    style(ax, f"HDD call price, {TICK:.0f} € per degree day", "Strike (HDD)", "Price (€)")
    save(fig, "call_price_vs_strike")

    # 6. backtest
    x = np.arange(len(bt))
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=SURFACE)
    ax.fill_between(x, bt["model_q05"], bt["model_q95"], color=BLUE, alpha=0.18, label="Model 90% interval")
    ax.plot(x, bt["model_mean"], color=BLUE, linewidth=2, label="Model mean")
    ax.plot(x, bt["burn_raw"], color=ORANGE, linewidth=2, label="Raw burn mean")
    ax.plot(x, bt["realised"], color=INK, marker="o", linewidth=0, markersize=7, label="Realised")
    ax.set_xticks(x, bt["season"], rotation=45, ha="right")
    ax.legend(frameon=False, labelcolor=INK, fontsize=9, ncol=2)
    style(ax, f"Out-of-sample forecasts: parameters frozen at {BACKTEST_CUTOFF}", "", "Winter HDD")
    save(fig, "backtest")

    if not synthetic:
        write_readme_results(results)
    print(json.dumps({k: v for k, v in results.items() if k != "backtest"}, indent=2))
    print(json.dumps({k: v for k, v in results["backtest"].items() if k != "table"}, indent=2))


def write_readme_results(r: dict, path: str = "README.md"):
    """Fill the results section of the README from real-data results (never from synthetic runs)."""
    m, p, b, d = r["model"], r["pricing"], r["backtest"], r["data"]
    block = f"""## Results (data {d['first_day']} to {d['last_day']}, {d['days']:,} days)

![HDD distribution](figures/hdd_distribution.png)

| Model | Value |
|---|---|
| Warming trend | {m['warming_trend_C_per_decade']:+.2f} °C per decade |
| AR order (BIC) | {m['ar_order_bic']} |
| Daily shock volatility, winter / summer | {m['daily_vol_winter_C']} / {m['daily_vol_summer_C']} °C |
| Lag-1 autocorrelation, before / after AR filter | {m['residual_acf_lag1_before']} / {m['residual_acf_lag1_after']} |
| Standardised shocks: skew / excess kurtosis | {m['z_skew']} / {m['z_excess_kurtosis']} |

**Winter {p['season']}**: expected HDD **{p['hdd_mean']:.0f}**, 90% range [{p['hdd_q05']:.0f}, {p['hdd_q95']:.0f}].
Call, strike {p['strike']:.0f}, {p['tick_eur']:.0f} € per HDD:

| Method | Price |
|---|---|
| **Model (Monte Carlo)** | **{p['call_price_eur']:,.0f} €** (± {1.96 * p['call_price_se_eur']:,.0f}) |
| Model with market price of risk λ = 0.05 | {p['call_price_lambda_0.05_eur']:,.0f} € |
| Burn analysis, raw | {p['burn_raw_price_eur']:,.0f} € |
| Burn analysis, detrended | {p['burn_detrended_price_eur']:,.0f} € |

![Call price vs strike](figures/call_price_vs_strike.png)

**Backtest** ({b['seasons']} winters, parameters frozen at {b['fit_until']}): realised HDD inside the model's
90% interval in {b['coverage_90']:.0%} of winters, mean PIT {b['mean_pit']:.2f} (0.5 if unbiased).
Forecast RMSE: model {b['rmse_model']:.0f} HDD, raw burn {b['rmse_burn_raw']:.0f}, detrended burn {b['rmse_burn_detrended']:.0f}.

![Backtest](figures/backtest.png)

<details><summary>Model diagnostics</summary>

![Temperature fit](figures/temperature_fit.png)
![Seasonal volatility](figures/seasonal_volatility.png)
![Residual diagnostics](figures/residual_diagnostics.png)
</details>"""
    text = Path(path).read_text()
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    i, j = text.index(start) + len(start), text.index(end)
    Path(path).write_text(text[:i] + "\n" + block + "\n" + text[j:])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    main(ap.parse_args().synthetic)
