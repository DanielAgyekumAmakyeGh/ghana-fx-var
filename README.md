# Ghana FX Value-at-Risk Engine

**Author:** Daniel Agyekum Amakye
**Email:** janetobosuayaa@gmail.com
**Role:** Quantitative Analyst — Risk Management

A quantitative risk project modelling a USD 20 million FX book at a
fictional Ghanaian bank, then calibrating and stress-testing the same
framework against **10 years of real Bank of Ghana USD/GHS data**.

**The bank is fictional. The methodology, the data, and the findings are real.**

---

## Two Calibrations

This repository contains two complementary VaR engines plus two
production-style reports:

### 1. Illustrative case (`src/ghana_fx_var.py`)

A clean teaching example with round numbers — a fictional bank
(Ghana Commercial & Investment Bank) holding USD 20m at 12.50 GHS/USD,
with volatility set to a plausible 1.2%/day.

### 2. Real-data calibration (`src/var_real_data.py`)

The same framework applied to **2,725 daily observations** of the
Bank of Ghana's USD/GHS reference rate (2015–2025).

### 3. GARCH(1,1) conditional VaR (`src/garch_var.py`)

A Student-t GARCH(1,1) fit that captures volatility clustering.

### 4. Daily FX risk report (`src/daily_risk_report.py`)

Production-style daily report using historical simulation as the
primary measure, with rolling and full-sample windows reported
side by side.

---

## Illustrative Case — Key Results

| Metric | Value |
|---|---|
| Position | USD 20,000,000 @ 12.50 |
| Position value | GHS 250,000,000 |
| Daily volatility | 1.20% |
| **Parametric VaR (99%, 1-day)** | **GHS 6,978,961** |
| Expected Shortfall | GHS 7,999,282 |
| 10-day Basel VaR | GHS 22,067,947 |

---

## Real-Data Calibration — Key Results

Calibrated against Bank of Ghana USD/GHS data, 2015-01-02 to 2025-12-31:

| Metric | Value |
|---|---|
| Observations | 2,725 |
| Latest USD/GHS | 10.45 |
| Position value | GHS 209,000,000 |
| Realised daily volatility | 0.7735% |
| Realised annual volatility | 12.28% |
| **Kurtosis** | **147.78** (normal = 0) |
| Skewness | −1.84 |
| Worst single day | −15.04% |

### Real-Data VaR (99%, 1-day)

| Method | VaR (GHS) | ES (GHS) |
|---|---|---|
| Parametric (full sample) | 3,760,816 | 4,308,633 |
| Historical (full sample) | 3,951,268 | 9,229,126 |
| **Ratio (Hist / Param)** | **1.05×** | **2.14×** |

### GARCH(1,1) Conditional VaR

| Parameter | Estimate | Interpretation |
|---|---|---|
| α₁ | 0.5287 | Strong reaction to yesterday's shock |
| β₁ | 0.4713 | Volatility persistence |
| α₁ + β₁ | **1.000** | **IGARCH — shocks are permanent** |
| ν (t d.o.f.) | 2.83 | Extreme fat tails |

| Measure | Value (GHS) |
|---|---|
| Full-sample daily vol (constant) | 0.7735% |
| **GARCH next-day conditional σ** | **1.7909%** |
| Parametric VaR (99%) | 3,760,816 |
| **GARCH VaR (99%)** | **8,707,564** |
| **GARCH / Parametric ratio** | **2.32×** |

### Historical VaR: full sample vs. rolling window

| Measure | Window | Historical VaR |
|---|---|---|
| Full-sample | 2,725 days (2015–2025) | **GH₵3,951,268** |
| Rolling 250-day | Last 250 trading days | **GH₵9,581,018** |
| **Recency uplift** | — | **2.43×** |

Both are correct. The full-sample figure is the long-run, unconditional
estimate used for capital planning; the rolling 250-day figure is the
recent conditional estimate used for daily limit monitoring. The
framework never presents either in isolation.

### Real-Data Kupiec Backtest

| Item | Value |
|---|---|
| Backtest window | Rolling 250-day |
| Observations | 2,475 |
| Expected breaches | 24.75 |
| **Observed breaches** | **60** |
| **LR statistic** | **36.27** |
| Critical value (95%) | 3.84 |
| **Verdict** | **REJECTED** |

**The parametric model fails the Kupiec backtest outright.** This is
the central finding of the project.

### Volatility Regimes

| Year | Daily vol | Annual vol | VaR (GHS) |
|---|---|---|---|
| 2015 | 0.8535% | 13.55% | 4,149,900 |
| 2016 | 0.1596% | 2.53% | 775,935 |
| 2017 | 0.3070% | 4.87% | 1,492,586 |
| 2018 | 0.1492% | 2.37% | 725,510 |
| 2019 | 0.2546% | 4.04% | 1,237,767 |
| 2020 | 0.1497% | 2.38% | 727,672 |
| 2021 | 0.0552% | 0.88% | 268,605 |
| **2022** | **1.9397%** | **30.79%** | **9,430,984** |
| 2023 | 0.9332% | 14.81% | 4,537,446 |
| 2024 | 0.3003% | 4.77% | 1,460,052 |
| 2025 | 0.9400% | 14.92% | 4,570,294 |

**Range: 0.88% to 30.79% annualised — a 35× spread.**

### Overnight Depreciation Stress Scenarios

| Shock | Shocked USD/GHS | Revaluation (GHS) | % of Position |
|---|---|---|---|
| 10% | 11.4950 | 20,900,000 | 10.00% |
| 20% | 12.5400 | 41,800,000 | 20.00% |
| **30%** | **13.5850** | **62,700,000** | **30.00%** |

---

## The Comparison — What the Two Calibrations Teach Us

| Question | Illustrative | Real-Data |
|---|---|---|
| Assumed / realised volatility | 1.20% / day | 0.77% / day |
| VaR (99%, 1-day) | GHS 6.98m | GHS 3.76m |
| Fat tails captured? | No (assumed normal) | Yes (kurtosis 148) |
| Kupiec verdict | N/A | **REJECTED** |
| Model survives? | Unknown | **No** |

The illustrative case shows the **mechanics**. The real-data case shows
the **failure mode**.

---

## Methodology

### Parametric VaR
```
VaR = V × σ × z_α
```

### Expected Shortfall (normal case)
```
ES = V × σ × φ(z_α) / (1 − α)
```

### Historical Simulation VaR
```
VaR = empirical quantile of (−returns × V) at level α
```

### GARCH(1,1)
```
σ²_t = ω + α₁ ε²_{t−1} + β₁ σ²_{t−1}
```
with Student-t errors.

### Kupiec POF statistic
```
LR = −2 ln [ (1−p)^(n−x) p^x / ((1−x/n)^(n−x) (x/n)^x) ]  ~ χ²(1)
```
where `p = 1 − α` (expected breach rate).

---

## Installation

```bash
git clone https://github.com/DanielAgyekumAmakyeGh/ghana-fx-var.git
cd ghana-fx-var
pip install -r requirements.txt
```

## Usage

### Illustrative engine
```bash
python src/ghana_fx_var.py
```

### Real-data engine
```bash
python src/load_bog_data.py    # clean the raw BoG data
python src/var_real_data.py    # run the VaR analysis
```

### GARCH(1,1) conditional VaR
```bash
python src/garch_var.py
```

### Daily FX risk report (historical simulation primary)
```bash
python src/daily_risk_report.py
```

---

## Data & Assumptions

### Illustrative engine
- The bank (GCIB) is fictional
- Volatility, spot, and correlations are illustrative
- Purpose: demonstrate the mechanics cleanly

### Real-data engine
- **Source:** Bank of Ghana interbank FX rates
- **Range:** 2015-01-02 to 2025-12-31
- **Observations:** 2,725 daily mid rates
- **Raw file:** `data/usd_ghs_rates.csv`
- **Cleaned file:** `data/usd_ghs_clean.csv`

Both engines use the same methodology. Only the calibration differs.

---

## Limitations

- Normality assumed in the parametric model — falsified by the
  kurtosis of 147.78
- Constant volatility in the parametric model — GARCH shows α + β = 1.000
- Linear positions — no options / convexity
- Square-root-of-time scaling assumes i.i.d. returns
- Fat tails break the Kupiec test — this is the central finding,
  not a bug

---

## Project Structure

```
ghana-fx-var/
├── src/
│   ├── ghana_fx_var.py          # illustrative engine
│   ├── load_bog_data.py         # BoG data loader
│   ├── var_real_data.py         # real-data VaR engine
│   ├── garch_var.py             # GARCH(1,1) conditional VaR
│   └── daily_risk_report.py     # daily risk report
├── data/
│   ├── README.md
│   ├── usd_ghs_rates.csv        # raw BoG data
│   └── usd_ghs_clean.csv        # cleaned
├── docs/
│   ├── GCIB_Market_Risk_Report.tex
│   └── Model_Validation_Memo.tex
├── examples/
│   ├── sample_output.txt
│   ├── real_data_output.txt
│   ├── garch_output.txt
│   └── daily_risk_report_output.txt
├── CITATION.cff
├── CHANGELOG.md
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

---

## License

MIT License — see `LICENSE` for details.

## Contact

**Daniel Agyekum Amakye**
Quantitative Analyst — Risk Management
Email: janetobosuayaa@gmail.com