# CLAUDE.md

## What this project is

**fin-crypto-residual-momentum**: cross-sectional crypto momentum after
stripping BTC beta. Ranks coins by cumulative residual return (alpha
relative to BTC), not raw return. Sibling to `fin-crypto-lab`, which
provides the data pipeline, panel builder, backtest engine, and honesty
battery.

## Stack

- Python 3.12+, numpy, polars, uv-managed.
- Depends on `fin-crypto-lab` as a path dependency (`../fin-crypto-lab`).
- Uses `fin-crypto-lab`'s data (spot + futures OHLCV parquets), panel,
  backtest engine, universe builder, and overfit metrics verbatim.
- Only new code lives here: the extended signal and the sweep grid.

## Key files

| File | Purpose |
|---|---|
| `src/fin_crypto_residual_momentum/signals.py` | `residual_momentum()` and `raw_momentum()` |
| `src/fin_crypto_residual_momentum/sweep.py` | Grids for both signals: lookbacks (90, 180, 365) x top (10, 20, 30) x instruments |
| `src/fin_crypto_residual_momentum/run_sweep.py` | Sweep runner with honesty battery and head-to-head comparison mode |

## Commands

Run a single signal sweep:
```
uv run python -m fin_crypto_residual_momentum --instrument spot --signal residual
uv run python -m fin_crypto_residual_momentum --instrument spot --signal raw
```

Run head-to-head comparison (both signals, then side-by-side):
```
uv run python -m fin_crypto_residual_momentum --instrument spot --signal compare
uv run python -m fin_crypto_residual_momentum --instrument futures --signal compare
```

Run tests:
```
uv run pytest tests/ -v
```

## Key invariants

- Same invariants as `fin-crypto-lab`: 24/7 calendar, Sunday formations,
  next-session fills, same cost model and train/test split.
- Pre-registration discipline applies. Do not change grid or thresholds
  after seeing results.
- `beta_window` parameter: when set, OLS beta is estimated on the trailing
  N days but residuals are computed over the full lookback window.
