# fin-crypto-residual-momentum

Cross-sectional momentum after removing BTC beta. This strategy ranks coins by cumulative residual return (alpha relative to BTC), not raw return.

## Why residual momentum

Standard momentum in crypto includes BTC beta noise. When BTC rallies, everything rallies. Raw momentum ranks the coins with the most BTC exposure, not the coins with the strongest idiosyncratic performance. Residual momentum removes this bias.

Academic basis: Blitz and Vidojevic (2021) document the idiosyncratic momentum anomaly in equities. The same logic applies to crypto, where BTC is the dominant shared cause of price movement.

## Signal construction

For each coin on each formation date:

1. Get the trailing `L` days of daily returns for the coin and BTC.
2. Operate OLS: `r_i = alpha + beta * r_BTC + epsilon`.
3. Signal = sum of residuals (cumulative idiosyncratic return).
4. Rank coins by signal. Go long the top N, short the bottom N (or long-only top N on spot).

## Sweep grid

Lookbacks: 90, 180, 365 days. Portfolio dimensions: top 10, 20, 30. Instruments: spot and futures. A `beta_window` parameter controls the trailing window for the OLS beta estimate. The strategy calculates residuals on the full lookback window.

## Honesty battery

Same battery as `fin-crypto-lab`: DSR, PBO, train/test Sharpe, kill conditions. The grid is larger than `fin-crypto-lab` (adds beta-window variants), so the multiple-testing correction penalty is higher.

## Key risk

With only 90 days of daily returns, the OLS beta estimate has high noise. Coins with low liquidity produce unreliable betas and noisy residuals. A minimum-liquidity filter is necessary.

## Dependencies

This project depends on `fin-crypto-lab` as a path dependency (`../fin-crypto-lab`). Clone `fin-crypto-lab` as a sibling directory before you use this project.

## Source

Blitz, D. and Vidojevic, M. (2021). "The Idiosyncratic Momentum Anomaly." https://doi.org/10.2139/ssrn.3838856
