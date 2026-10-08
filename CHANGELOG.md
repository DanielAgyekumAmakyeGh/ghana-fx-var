# Changelog

## [1.0.0] — 2026-10-08

### Added
- Illustrative VaR engine (`src/ghana_fx_var.py`)
- Real-data VaR engine (`src/var_real_data.py`)
- GARCH(1,1) conditional VaR engine (`src/garch_var.py`)
- Daily FX risk report — historical simulation primary (`src/daily_risk_report.py`)
- BoG data loader (`src/load_bog_data.py`)
- 10 years of Bank of Ghana USD/GHS data (2015–2025, 2,725 observations)
- Rolling-window Kupiec backtest
- Overnight depreciation stress scenarios (10%, 20%, 30%)
- LaTeX technical report (`docs/GCIB_Market_Risk_Report.tex`)
- Model validation memo (`docs/Model_Validation_Memo.tex`)

### Findings
- Kurtosis 147.78 rejects normality
- Parametric VaR fails Kupiec (LR = 36.27, 60 breaches vs. 24.75 expected)
- Historical ES is 2.14× the parametric ES
- GARCH α + β = 1.000 (IGARCH); conditional VaR is 2.32× parametric
- Volatility regime range 0.88% – 30.79% annualised (35× spread)
- Recency uplift (rolling 250d vs. full sample) is 2.43×