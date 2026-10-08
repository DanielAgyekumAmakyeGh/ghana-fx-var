"""
GARCH(1,1) VaR engine for USD/GHS
GCIB | Quantitative Risk & Financial Engineering
Author: Daniel Agyekum Amakye
"""

from pathlib import Path
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm


DATA_PATH = Path("data/usd_ghs_clean.csv")
POSITION_USD = 20_000_000


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    latest_rate = df["Mid Rate"].iloc[-1]
    position_value = POSITION_USD * latest_rate
    returns = df["log_return"].dropna() * 100  # arch wants pct returns

    # Fit GARCH(1,1)
    model = arch_model(returns, vol="Garch", p=1, q=1, dist="t")
    res = model.fit(disp="off")

    # Forecast next-day volatility
    forecast = res.forecast(horizon=1)
    next_sigma = np.sqrt(forecast.variance.values[-1, 0]) / 100

    z = norm.ppf(0.99)
    var = position_value * next_sigma * z

    print("=" * 60)
    print("GARCH(1,1) VaR - USD/GHS")
    print("=" * 60)
    print(f"Latest rate              : {latest_rate:.4f}")
    print(f"Position value           : GHS {position_value:,.2f}")
    print()
    print(f"GARCH next-day sigma     : {next_sigma * 100:.4f}%")
    print(f"GARCH VaR 99% (1-day)    : GHS {var:,.2f}")
    print()
    print("Model summary:")
    print(res.summary())


if __name__ == "__main__":
    main()