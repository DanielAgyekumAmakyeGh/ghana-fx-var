"""
GCIB - Quantitative Risk & Financial Engineering
Author  : Daniel Agyekum Amakye
Email   : janetobosuayaa@gmail.com
Role    : Quantitative Analyst - Risk Management
Module  : Daily FX Risk Report - Historical Simulation as Primary Measure

This script implements Recommendation 1 of the technical report:
historical simulation is used as the PRIMARY daily VaR measure for
the FX book, with parametric and GARCH VaR as secondary checks.

Both the rolling 250-day window and the full-sample window are
reported side by side. The framework never presents either in
isolation.
"""

from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------- config
DATA_PATH = Path("data/usd_ghs_clean.csv")
POSITION_USD = 20_000_000
WINDOW = 250
ALPHA = 0.99
TRADING_DAYS = 252


# ---------------------------------------------------------------- core
def historical_var(losses, alpha):
    return float(np.percentile(losses, alpha * 100))


def historical_es(losses, alpha):
    var = historical_var(losses, alpha)
    tail = losses[losses >= var]
    return float(tail.mean()) if len(tail) else var


def parametric_var(losses, alpha):
    sigma = losses.std()
    z = norm.ppf(alpha)
    return float(sigma * z)


# ---------------------------------------------------------------- report
def build_report(df, position_usd=POSITION_USD, window=WINDOW, alpha=ALPHA):
    latest_rate = float(df["Mid Rate"].iloc[-1])
    position_value = position_usd * latest_rate
    returns = df["log_return"].dropna()

    # --- Full-sample historical VaR (long-run view) ---
    full_losses = -returns * position_value
    full_var = historical_var(full_losses, alpha)
    full_es = historical_es(full_losses, alpha)

    # --- Rolling-window historical VaR (recent regime) ---
    recent = returns.iloc[-window:]
    recent_losses = -recent * position_value
    recent_var = historical_var(recent_losses, alpha)
    recent_es = historical_es(recent_losses, alpha)
    breaches = int((recent_losses > recent_var).sum())

    # --- Parametric VaR on the same rolling window ---
    param_var = parametric_var(recent_losses, alpha)

    return {
        "as_of": df["Date"].iloc[-1].date(),
        "latest_rate": latest_rate,
        "position_usd": position_usd,
        "position_value": position_value,
        "window": window,
        "n_full": len(returns),
        "alpha": alpha,
        "full_var": full_var,
        "full_es": full_es,
        "recent_var": recent_var,
        "recent_es": recent_es,
        "recent_vol": recent.std(),
        "recent_annual_vol": recent.std() * np.sqrt(TRADING_DAYS),
        "param_var": param_var,
        "breaches": breaches,
    }


def print_report(r):
    line = "=" * 70
    print(line)
    print(f" GCIB | Daily FX Risk Report        | {r['as_of']}")
    print(line)
    print(f" Currency pair        : USD/GHS")
    print(f" Spot rate            : {r['latest_rate']:.4f}")
    print(f" Position             : USD {r['position_usd']:,}")
    print(f" Position value       : GHS {r['position_value']:,.2f}")
    print(f" Confidence           : {r['alpha'] * 100:.2f}%")
    print(line)
    print()
    print(" HISTORICAL SIMULATION VaR (primary measure)")
    print(f"   Rolling {r['window']}-day window    : GHS {r['recent_var']:>15,.2f}")
    print(f"   Full sample ({r['n_full']} days)   : GHS {r['full_var']:>15,.2f}")
    print(f"   Recency uplift        : {r['recent_var'] / r['full_var']:.3f}x")
    print()
    print(" EXPECTED SHORTFALL")
    print(f"   Rolling {r['window']}-day window    : GHS {r['recent_es']:>15,.2f}")
    print(f"   Full sample           : GHS {r['full_es']:>15,.2f}")
    print()
    print(" SECONDARY CHECK (parametric on rolling window)")
    print(f"   Parametric VaR        : GHS {r['param_var']:>15,.2f}")
    print(f"   Parametric vs Hist    : {r['param_var'] / r['recent_var']:.3f}x")
    print()
    print(" RECENT REGIME")
    print(f"   Realised vol (ann.)   : {r['recent_annual_vol'] * 100:.4f}%")
    print(f"   Breaches in window    : {r['breaches']}")
    print(f"   Expected (1% of {r['window']}) : {r['window'] * (1 - r['alpha']):.2f}")
    print(line)


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    r = build_report(df)
    print_report(r)


if __name__ == "__main__":
    main()