"""
GCIB - Quantitative Risk & Financial Engineering
Author  : Daniel Agyekum Amakye
Email   : janetobosuayaa@gmail.com
Role    : Quantitative Analyst - Risk Management
Module  : Real-Data FX VaR Engine (Bank of Ghana USD/GHS, 2015-2025)

Sections:
  - Full-sample parametric and historical VaR
  - Regime-conditional VaR by year
  - Rolling-window Kupiec backtest
  - Backward-looking stress: 2022 crisis volatility
  - Forward-looking stress: overnight depreciation scenarios
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm


# ---------------------------------------------------------------- config
DATA_PATH = Path("data/usd_ghs_clean.csv")
POSITION_USD = 20_000_000
ALPHA = 0.99
TRADING_DAYS = 252
ROLLING_WINDOW = 250
DEPRECIATION_SHOCKS = (0.10, 0.20, 0.30)


# ---------------------------------------------------------------- load
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------- VaR
def parametric_var(returns, position_value, alpha=ALPHA, horizon=1):
    sigma = returns.std() * np.sqrt(horizon)
    z = norm.ppf(alpha)
    var = position_value * sigma * z
    es = position_value * sigma * norm.pdf(z) / (1 - alpha)
    return {"var": var, "es": es, "sigma": sigma, "z": z}


def historical_var(returns, position_value, alpha=ALPHA):
    losses = -returns * position_value
    var = np.percentile(losses, alpha * 100)
    tail = losses[losses >= var]
    es = tail.mean() if len(tail) > 0 else var
    return {"var": var, "es": es}


# ---------------------------------------------------------------- Kupiec
def kupiec_pof(n, alpha, x):
    p = 1 - alpha
    if x == 0:
        return -2 * n * np.log(1 - p)
    phat = x / n
    return -2 * (
        (n - x) * np.log(1 - p) + x * np.log(p)
        - (n - x) * np.log(1 - phat) - x * np.log(phat)
    )


def rolling_backtest(returns, position_value,
                     alpha=ALPHA, window=ROLLING_WINDOW):
    losses = -returns * position_value
    breaches = 0
    total = 0
    for i in range(window, len(losses)):
        past = losses.iloc[i - window:i]
        var_t = np.percentile(past, alpha * 100)
        if losses.iloc[i] > var_t:
            breaches += 1
        total += 1
    return breaches, total


# ---------------------------------------------------------------- stress
def depreciation_scenarios(latest_rate, position_value,
                           shocks=DEPRECIATION_SHOCKS):
    rows = []
    for shock in shocks:
        shocked_rate = latest_rate * (1 + shock)
        revaluation = POSITION_USD * (shocked_rate - latest_rate)
        rows.append({
            "shock": shock,
            "shocked_rate": shocked_rate,
            "revaluation": revaluation,
            "pct_of_position": revaluation / position_value,
        })
    return rows


# ---------------------------------------------------------------- sections
def print_header(df, returns, latest_rate, position_value):
    print("=" * 72)
    print("GCIB | Real-Data FX VaR Engine - Bank of Ghana USD/GHS")
    print("=" * 72)
    print(f"Data range         : {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Observations       : {len(returns)}")
    print(f"Latest USD/GHS     : {latest_rate:.4f}")
    print(f"Position           : USD {POSITION_USD:,}")
    print(f"Position value     : GHS {position_value:,.2f}")
    print()


def print_full_sample(returns, position_value):
    print("--- FULL-SAMPLE (2015-2025) ---")
    p = parametric_var(returns, position_value)
    h = historical_var(returns, position_value)
    print(f"Realised daily vol : {returns.std() * 100:.4f}%")
    print(f"Realised annual vol: {returns.std() * np.sqrt(TRADING_DAYS) * 100:.4f}%")
    print(f"Kurtosis           : {returns.kurtosis():.4f}  (normal = 0)")
    print()
    print(f"Parametric VaR 99% : GHS {p['var']:>15,.2f}  (full sample)")
    print(f"Parametric ES  99% : GHS {p['es']:>15,.2f}  (full sample)")
    print(f"Historical VaR 99% : GHS {h['var']:>15,.2f}  (full sample, {len(returns)} days)")
    print(f"Historical ES  99% : GHS {h['es']:>15,.2f}  (full sample)")
    print(f"Parametric vs Hist.: {p['var'] / h['var']:.3f}x")
    return p


def print_regime_var(df_year, position_value):
    print("\n--- REGIME-CONDITIONAL VaR (99%, 1-day) ---")
    print(f"{'Year':<6} {'Days':<6} {'Daily vol':<12} {'VaR (GHS)':<18} {'ES (GHS)':<18}")
    print("-" * 72)
    for year in sorted(df_year["year"].unique()):
        y_returns = df_year.loc[df_year["year"] == year, "log_return"].dropna()
        if len(y_returns) > 20:
            p_y = parametric_var(y_returns, position_value)
            print(f"{year:<6} {len(y_returns):<6} "
                  f"{y_returns.std() * 100:>8.4f}%    "
                  f"{p_y['var']:>15,.2f}    {p_y['es']:>15,.2f}")


def print_kupiec(returns, position_value):
    print("\n--- KUPIEC BACKTEST (rolling 250-day window, 99%) ---")
    breaches, total = rolling_backtest(returns, position_value)
    expected = total * (1 - ALPHA)
    lr = kupiec_pof(total, ALPHA, breaches)
    crit = chi2.ppf(0.95, df=1)
    verdict = "ACCEPTED" if lr < crit else "REJECTED"
    print(f"Backtest window    : rolling {ROLLING_WINDOW} days")
    print(f"Observations       : {total}")
    print(f"Expected breaches  : {expected:.2f}")
    print(f"Observed breaches  : {breaches}")
    print(f"Breach rate        : {breaches / total * 100:.3f}%")
    print(f"LR statistic       : {lr:.4f}")
    print(f"Critical (95%)     : {crit:.4f}")
    print(f"Verdict            : {verdict}")


def print_crisis_stress(df_year, position_value, p_full_sample):
    print("\n--- STRESS: 2022 crisis volatility applied to today ---")
    crisis = df_year.loc[df_year["year"] == 2022, "log_return"].dropna()
    p_crisis = parametric_var(crisis, position_value)
    print(f"2022 daily vol     : {crisis.std() * 100:.4f}%")
    print(f"Stressed VaR 99%   : GHS {p_crisis['var']:>15,.2f}")
    print(f"Stressed ES  99%   : GHS {p_crisis['es']:>15,.2f}")
    ratio = p_crisis["var"] / p_full_sample["var"]
    print(f"Stress / calm ratio: {ratio:.2f}x")


def print_depreciation_stress(latest_rate, position_value):
    print("\n--- STRESS: overnight GHS depreciation scenarios ---")
    print(f"{'Shock':<8} {'Shocked rate':<14} "
          f"{'Revaluation (GHS)':<22} {'% of position':<14}")
    print("-" * 72)
    for s in depreciation_scenarios(latest_rate, position_value):
        print(f"{s['shock'] * 100:>5.0f}%   {s['shocked_rate']:>10.4f}     "
              f"{s['revaluation']:>18,.2f}     {s['pct_of_position'] * 100:>6.2f}%")
    print()
    print("Note: FX revaluation on the long USD 20m position; the funding")
    print("side of a GHS-funded book bears the symmetric loss.")


# ---------------------------------------------------------------- main
def main():
    df = load_data()
    latest_rate = df["Mid Rate"].iloc[-1]
    position_value = POSITION_USD * latest_rate
    returns = df["log_return"].dropna()

    df_year = df.copy()
    df_year.loc[:, "year"] = df_year["Date"].dt.year

    print_header(df, returns, latest_rate, position_value)
    p = print_full_sample(returns, position_value)
    print_regime_var(df_year, position_value)
    print_kupiec(returns, position_value)
    print_crisis_stress(df_year, position_value, p)
    print_depreciation_stress(latest_rate, position_value)

    print("=" * 72)


if __name__ == "__main__":
    main()