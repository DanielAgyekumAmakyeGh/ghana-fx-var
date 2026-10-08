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

This repository contains two complementary VaR engines:

### 1. Illustrative case (`src/ghana_fx_var.py`)

A clean teaching example with round numbers — a fictional bank
(Ghana Commercial & Investment Bank) holding USD 20m at 12.50 GHS/USD,
with volatility set to a plausible 1.2%/day.

**Purpose:** demonstrates the mechanics — parametric VaR, Expected
Shortfall, Kupiec backtesting, and portfolio VaR — in a controlled
setting.

### 2. Real-data calibration (`src/var_real_data.py`)

The same framework applied to **2,725 daily observations** of the
Bank of Ghana's USD/GHS reference rate (2015–2025).

**Purpose:** tests whether the model survives real Ghanaian market
data — including the 2022 cedi crisis.

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

Calibrated against Bank of Ghana USD/GHS data, 2015-01-01 to 2025-12-31:

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
| Parametric | 3,760,816 | 4,308,633 |
| Historical | 3,951,268 | 9,229,126 |
| **Ratio (Hist / Param)** | **1.05×** | **2.14×** |

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

### Real-Data Volatility Regimes

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

---

## The Comparison — What the Two Calibrations Teach Us

| Question | Illustrative | Real-Data |
|---|---|---|
| Assumed volatility | 1.20% / day | 0.77% / day |
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

### Kupiec POF statistic
```
LR = −2 ln [ (1−p)^(n−x) p^x / ((1−x/n)^(n−x) (x/n)^x) ]  ~ χ²(1)
```
where `p = 1 − α` (expected breach rate).

### Portfolio VaR
```
σ_P = sqrt(wᵀ Σ w)
VaR_P = z_α × σ_P
```

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

---

## Data & Assumptions

### Illustrative engine
- The bank (GCIB) is fictional
- Volatility, spot, and correlations are illustrative
- Purpose: demonstrate the mechanics cleanly

### Real-data engine
- **Source:** Bank of Ghana interbank FX rates
- **Range:** 2015-01-01 to 2025-12-31
- **Observations:** 2,725 daily mid rates
- **Raw file:** `data/usd_ghs_rates.csv`
- **Cleaned file:** `data/usd_ghs_clean.csv`

Both engines use the same methodology. Only the calibration differs.

---

## Limitations

- Normality assumed in the parametric model — falsified by the
  kurtosis of 147.78
- Constant volatility — no GARCH clustering
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
│   └── var_real_data.py         # real-data VaR engine
├── data/
│   ├── README.md
│   ├── usd_ghs_rates.csv        # raw BoG data
│   └── usd_ghs_clean.csv        # cleaned
├── examples/
│   ├── sample_output.txt        # illustrative output
│   └── real_data_output.txt     # real-data output
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