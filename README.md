# Ghana FX Value-at-Risk Engine
"""
GCIB - Quantitative Risk & Financial Engineering
Author  : Daniel Agyekum Amakye
Email   : janetobosuayaa@gmail.com
Role    : Quantitative Analyst - Risk Management
Module  : Ghana FX VaR Engine + Kupiec Backtest + Portfolio VaR
"""

## Overview

A self-directed quantitative risk project modelling a USD 20 million FX
book at a fictional Ghanaian bank (Ghana Commercial & Investment Bank — GCIB).

The project demonstrates a full market risk stack:

- Parametric Value-at-Risk (VaR)
- Expected Shortfall (ES)
- Dual-regime loss distribution analysis (calm vs. 2022 cedi crisis)
- Kupiec Proportion-of-Failures (POF) backtesting
- Multi-currency portfolio VaR (USD, EUR, GBP)
- Basel III / Bank of Ghana regulatory capital mapping

**The bank is fictional. The methodology and regulatory framework are real.**

---

## Key Findings

| Metric | Value |
|---|---|
| 1-day 99% VaR (calm regime)  | GH₵ 6,979,044 |
| 10-day 99% VaR (Basel)       | GH₵ 22,069,674 |
| Expected Shortfall (1-day)   | GH₵ 7,995,643 |
| 1-day 99% VaR (2022 crisis)  | GH₵ 16,486,441 |
| Portfolio VaR (USD/EUR/GBP)  | GH₵ 11,529,764 |
| Diversification benefit      | 9.6% |

**Headline insight:** A VaR model calibrated on calm 2024 data understates
cedi tail risk by ~2.5x during a depreciation shock. The model fails the
Kupiec POF test outright on 2022 data (9 breaches vs. 2.5 expected).

---

## Installation

```bash
git clone https://github.com/YOUR-USERNAME/ghana-fx-var.git
cd ghana-fx-var
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- `numpy`
- `scipy`

---

## Usage

```bash
python src/ghana_fx_var.py
```

The script prompts for:

- Currency pair (e.g. `USD/GHS`)
- Spot rate
- Position size (base currency)
- Daily volatility (%)
- Confidence level (%)

It then reports:

1. 1-day and 10-day VaR
2. Expected Shortfall
3. Stress-test results across Ghanaian volatility regimes (2022 / 2023 / 2024)
4. Kupiec backtest on three GHS market windows
5. Multi-currency portfolio VaR and diversification benefit

### Sample run

```
--- SINGLE-POSITION VaR ---
   1-day VaR (99%) : GHS       6,979,043.62
  10-day VaR (99%) : GHS      22,069,673.74
   ES   1-day (99%) : GHS       7,995,642.66

--- CRISIS REGIME STRESS (Ghana 2022) ---
       calm_2024 (vol 1.13%): VaR = GHS    6,594,576.36
     adjust_2023 (vol 1.76%): VaR = GHS   10,258,229.90
     crisis_2022 (vol 2.83%): VaR = GHS   16,486,440.90

--- KUPIEC BACKTEST (illustrative GHS windows) ---
       Calm 2024:  2 breaches, LR =  0.108  -> ACCEPTED
     Adjust 2023:  4 breaches, LR =  0.769  -> ACCEPTED
     Crisis 2022:  9 breaches, LR = 10.229  -> REJECTED

--- MULTI-CURRENCY PORTFOLIO VaR (Ghana book) ---
  Portfolio sigma   : GHS    4,956,165.07
  Portfolio VaR 99% : GHS   11,529,764.09
  Sum of standalone : GHS   12,756,528.57
  Diversification   : GHS    1,226,764.48 (9.6%)
```

---

## Methodology

### Parametric VaR

```
VaR = V * sigma * z_alpha
```

Where `V` is position value in GHS, `sigma` is daily volatility, and
`z_alpha` is the standard normal quantile at the chosen confidence level.

### Expected Shortfall (normal case)

```
ES = V * sigma * phi(z_alpha) / (1 - alpha)
```

### Kupiec POF statistic

```
LR_POF = -2 ln [ (1-p)^(n-x) p^x / ((1-x/n)^(n-x) (x/n)^x) ]  ~ chi2(1)
```

Where `p = 1 - alpha` (expected breach rate) and `x` is observed breaches.

### Portfolio VaR

```
sigma_P = sqrt(w.T * Sigma * w)
VaR_P   = z_alpha * sigma_P
```

---

## Ghana Market Context

| Period | Annual vol | Daily vol | Context |
|---|---|---|---|
| 2019 | 8%  | 0.50% | Pre-COVID stability |
| 2020 | 15% | 0.95% | Pandemic shock |
| 2022 | 45% | 2.83% | Cedi depreciation, IMF programme |
| 2023 | 28% | 1.76% | Post-restructuring |
| 2024 | 18% | 1.13% | Stabilisation, BoG tightening |

Structural features of the USD/GHS market:

- Thin interbank liquidity — spreads widen 10x under stress
- Bank of Ghana intervention via FX auctions
- High correlation across GHS crosses (common cedi factor)
- Fat-tailed returns (kurtosis well above 3)

---

## Regulatory Framework

- **Basel III** Internal Models Approach (IMA)
- **Bank of Ghana** minimum CAR: 13% (above Basel III's 8%)
- **BoG reporting:** FX Position Return (daily), Market Risk Return (monthly),
  CAR Return BSD 4 (monthly), ICAAP (annual)

---

## Data & Assumptions

This is a **case study**, not a production risk system.

- **The bank (GCIB) is fictional.**
- Volatility figures, correlations, and spot rates reflect publicly observable
  behaviour of the USD/GHS pair (Bank of Ghana publications, 2019-2024) and
  are rounded for illustration.
- The purpose is to demonstrate **method** — VaR construction, ES, Kupiec
  backtesting, portfolio aggregation, and Basel III / BoG capital mapping —
  not to publish a live risk number.

### Replacing with real data

1. Download USD/GHS daily rates from the Bank of Ghana or a market data provider.
2. Replace the volatility input with the realised standard deviation of log returns.
3. Rerun the script — the framework is data-source agnostic.

---

## Limitations

- Normality assumed — GHS returns exhibit fat tails
- Constant volatility — no GARCH clustering
- Linear positions — no options/convexity
- Square-root-of-time scaling assumes i.i.d. returns
- Liquidity assumed sufficient at market prices
- Illustrative rather than live data

---

## Project Structure

```
ghana-fx-var/
├── src/
│   └── ghana_fx_var.py        # main VaR engine
├── data/
│   └── README.md              # data sources and assumptions
├── examples/
│   └── sample_output.txt      # sample console output
├── docs/                      # (optional) LaTeX report
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

---

## License

MIT License — see `LICENSE` for details.

---
## Contact

**Daniel Agyekum Amakye**
Quantitative Analyst — Risk Management
📧 janetobosuayaa@gmail.com