"""Residual momentum signal with configurable beta estimation window."""
import numpy as np

from fin_crypto_lab.panel import Panel


def residual_momentum(
    panel: Panel, f_idx: int, lookback: int = 365,
    skip: int = 7, market_col: int | None = None,
    beta_window: int | None = None,
) -> np.ndarray:
    """Momentum after stripping market (BTC) beta via OLS.

    beta_window: number of trailing days for OLS regression. Defaults to
    lookback (full window). Use shorter values (60, 90, 120) to test
    sensitivity of beta estimation noise.

    Signal = cumulative residual return over the full lookback window,
    using beta estimated from the trailing beta_window days.
    """
    if beta_window is None:
        beta_window = lookback
    if f_idx < lookback:
        raise ValueError(f"f_idx {f_idx} < lookback {lookback}")

    start = f_idx - lookback
    end = f_idx - skip
    closes = panel.close_ff[start:end + 1, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        rets = closes[1:] / closes[:-1] - 1.0

    if market_col is not None:
        mkt = rets[:, market_col].copy()
    else:
        mkt = np.nanmean(rets, axis=1)

    beta_start = max(0, rets.shape[0] - beta_window)
    valid_mkt = np.isfinite(mkt)
    n = rets.shape[1]
    out = np.full(n, np.nan)

    for j in range(n):
        col = rets[:, j]

        # estimate beta on the trailing beta_window
        bmask = valid_mkt.copy()
        bmask[:beta_start] = False
        bmask &= np.isfinite(col)
        if bmask.sum() < 20:
            continue
        by, bx = col[bmask], mkt[bmask]
        bx_dm = bx - bx.mean()
        denom = np.dot(bx_dm, bx_dm)
        beta = np.dot(bx_dm, by) / denom if denom > 1e-15 else 0.0
        alpha = by.mean() - beta * bx.mean()

        # compute residuals over the full lookback window
        full_mask = valid_mkt & np.isfinite(col)
        if full_mask.sum() < 20:
            continue
        fy, fx = col[full_mask], mkt[full_mask]
        resid = fy - alpha - beta * fx
        out[j] = float(np.sum(resid))

    return out
