"""
GCIB - Quantitative Risk & Financial Engineering
Author  : Daniel Agyekum Amakye
Email   : janetobosuayaa@gmail.com
Role    : Quantitative Analyst - Risk Management
Module  : Ghana FX VaR Engine + Kupiec Backtest + Portfolio VaR
"""

import numpy as np
from scipy.stats import norm, chi2


# ---- Ghana volatility regimes (annualised -> daily) ----
GHS_REGIMES = {
    "calm_2024":     0.18 / np.sqrt(252),   # ~1.13%/day
    "adjust_2023":   0.28 / np.sqrt(252),   # ~1.76%/day
    "crisis_2022":   0.45 / np.sqrt(252),   # ~2.83%/day
}


def get_float(prompt, minimum=None, maximum=None):
    while True:
        raw = input(prompt).strip().replace(",", "").replace("_", "")
        try:
            value = float(raw)
        except ValueError:
            print("  ! Enter a valid number.\n"); continue
        if minimum is not None and value <= minimum:
            print(f"  ! Must be > {minimum}.\n"); continue
        if maximum is not None and value >= maximum:
            print(f"  ! Must be < {maximum}.\n"); continue
        return value


def collect_inputs():
    print("--- INPUTS (Ghana FX Desk) ---")
    pair     = input("  Currency pair (default USD/GHS): ").strip() or "USD/GHS"
    spot     = get_float(f"  Spot rate ({pair})     : ", minimum=0)
    position = get_float("  Position (base ccy)   : ", minimum=0)
    vol_pct  = get_float("  Daily vol (%)         : ", minimum=0, maximum=100)
    conf_pct = get_float("  Confidence level (%)  : ", minimum=50, maximum=99.99)
    return {
        "pair": pair, "spot": spot, "position": position,
        "vol": vol_pct / 100.0, "alpha": conf_pct / 100.0,
    }


def compute_var(p, horizon=1):
    notional = p["position"] * p["spot"]
    z        = norm.ppf(p["alpha"])
    sigma_h  = p["vol"] * np.sqrt(horizon)
    var_ghs  = notional * sigma_h * z
    es_ghs   = notional * sigma_h * (norm.pdf(z) / (1 - p["alpha"]))
    return {"notional": notional, "z": z, "sigma_h": sigma_h,
            "var": var_ghs, "es": es_ghs}


def kupiec_pof(n, alpha, x):
    """Kupiec Proportion-of-Failures (POF) test.

    Parameters
    ----------
    n     : number of observations (e.g. 250 trading days)
    alpha : confidence level (e.g. 0.99)
    x     : observed number of VaR breaches

    Returns
    -------
    LR statistic (chi-square, df=1). Compare to chi2.ppf(0.95, df=1) = 3.841.
    """
    p    = 1 - alpha          # expected breach rate under the model
    phat = x / n              # observed breach rate

    if x == 0:
        # Avoid log(0)
        return -2 * n * np.log(1 - p)

    return -2 * (
        (n - x) * np.log(1 - p) + x * np.log(p)
        - (n - x) * np.log(1 - phat) - x * np.log(phat)
    )


def portfolio_var(positions, vols, corr_matrix, alpha=0.99):
    w = np.array(positions)
    s = np.array(vols)
    C = np.array(corr_matrix)
    Sigma = np.outer(s, s) * C
    sigma_P = np.sqrt(w @ Sigma @ w)
    z = norm.ppf(alpha)
    return sigma_P, z * sigma_P


def main():
    p = collect_inputs()

    print("\n--- SINGLE-POSITION VaR ---")
    for h in (1, 10):
        r = compute_var(p, horizon=h)
        print(f"  {h:>2}-day VaR (99%) : GHS {r['var']:>18,.2f}")
    r1 = compute_var(p, horizon=1)
    print(f"  ES   1-day (99%) : GHS {r1['es']:>18,.2f}")

    print("\n--- CRISIS REGIME STRESS (Ghana 2022) ---")
    for name, vol in GHS_REGIMES.items():
        stressed = dict(p); stressed["vol"] = vol
        r = compute_var(stressed, horizon=1)
        print(f"  {name:>14} (vol {vol*100:.2f}%): "
              f"VaR = GHS {r['var']:>15,.2f}")

    print("\n--- KUPIEC BACKTEST (illustrative GHS windows) ---")
    crit = chi2.ppf(0.95, df=1)
    for label, x in [("Calm 2024", 2), ("Adjust 2023", 4), ("Crisis 2022", 9)]:
        lr = kupiec_pof(250, p["alpha"], x)
        verdict = "ACCEPTED" if lr < crit else "REJECTED"
        print(f"  {label:>14}: {x:>2} breaches, LR = {lr:6.3f}  -> {verdict}")

    print("\n--- MULTI-CURRENCY PORTFOLIO VaR (Ghana book) ---")
    positions = [250_000_000, 136_000_000, 79_000_000]
    vols      = [0.012, 0.011, 0.0125]
    corr = [[1.00, 0.70, 0.65],
            [0.70, 1.00, 0.75],
            [0.65, 0.75, 1.00]]
    sigma_P, var_P = portfolio_var(positions, vols, corr, alpha=p["alpha"])
    var_sum = sum(norm.ppf(p["alpha"]) * pos * vol
                  for pos, vol in zip(positions, vols))
    print(f"  Portfolio sigma   : GHS {sigma_P:>15,.2f}")
    print(f"  Portfolio VaR 99% : GHS {var_P:>15,.2f}")
    print(f"  Sum of standalone : GHS {var_sum:>15,.2f}")
    print(f"  Diversification   : GHS {var_sum - var_P:>15,.2f} "
          f"({100*(var_sum - var_P)/var_sum:.1f}%)")


if __name__ == "__main__":
    main()