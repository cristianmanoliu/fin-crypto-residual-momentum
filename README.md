# fin-crypto-residual-momentum

Cross-sectional momentum after stripping BTC beta. Rank coins by cumulative
residual return (alpha relative to BTC), not raw return.

## Origin

Research idea #4 from `fin-crypto-lab` (2026-09-19). Standard momentum in
crypto is contaminated by BTC beta. When BTC rallies, everything rallies.
Raw momentum ranks "most exposed to BTC" rather than genuine idiosyncratic
strength. Residual momentum removes this confound.

## What we know

- Requires no new data. Uses the same spot (or futures) OHLCV from
  `fin-crypto-lab`.
- Implementation: rolling 90-day OLS regression of each pair's daily returns
  on BTC daily returns. Signal = cumulative residual return over the lookback
  window.
- Academic basis: Blitz & Vidojevic 2021 document the idiosyncratic momentum
  anomaly in equities. The same logic applies to crypto, where BTC is the
  dominant common factor.

## Signal construction

For each coin `i` on each formation date:

1. Take the trailing `L` days of daily returns for coin `i` and BTC.
2. Run OLS: `r_i = alpha + beta * r_BTC + epsilon`.
3. Signal = sum of residuals (cumulative idiosyncratic return).
4. Rank coins by signal. Go long the top N, short the bottom N (or long-only
   top N if shorting is not available on spot).

## What to do next

### Phase 1: Implementation

1. Add a `residual_momentum_signal()` function to the signal module. Inputs:
   returns DataFrame, BTC returns Series, lookback. Output: signal Series.
2. Run the same grid as `fin-crypto-lab`: lookbacks (90, 180, 365), portfolio
   sizes (top 10, 20, 30), spot + futures.
3. Apply the honesty battery.

### Phase 2: Comparison

4. Compare residual momentum to raw momentum head-to-head on the same
   universe and cost model. The value is in the DIFFERENCE.
5. If residual momentum passes and raw does not, the BTC-beta contamination
   hypothesis is confirmed.
6. Test a blend: 50% raw + 50% residual momentum.

### Phase 3: Robustness

7. Try different rolling windows for the beta estimate (60, 90, 120 days).
8. Try multi-factor residuals (BTC + ETH) instead of BTC-only.

## Honesty method

Same battery as `fin-crypto-lab`: DSR, PBO, train/test Sharpe, kill
conditions. The grid is larger (adds rolling window variants), so the
multiple-testing correction penalty is higher.

## Key risk

**Beta estimation noise.** With only 90 days of daily returns, the OLS beta
estimate is noisy. Coins with low liquidity will have unreliable betas,
producing noisy residuals. A minimum-liquidity filter is essential.

## Reference

Blitz & Vidojevic 2021: "The Idiosyncratic Momentum Anomaly"
(https://doi.org/10.2139/ssrn.3838856).
