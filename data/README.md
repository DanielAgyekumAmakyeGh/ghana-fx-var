# Data Sources

This project uses illustrative market parameters calibrated to publicly
observable behaviour of the USD/GHS pair.

## Reference sources

- Bank of Ghana - Monetary Policy Reports, Statistical Bulletins
- Bank of Ghana - Daily USD/GHS reference rate publications
- IMF - Ghana Article IV consultations

## Real data

To substitute real data:

1. Download daily USD/GHS rates (BoG or market data provider).
2. Compute log returns: r_t = ln(P_t / P_{t-1}).
3. Use the standard deviation of r_t as daily volatility.
4. Plug into src/ghana_fx_var.py.

The framework does not depend on any specific data source.