"""Momentum signals: raw and residual (BTC-beta-stripped)."""
import numpy as np

from fin_crypto_lab.panel import Panel
from fin_crypto_lab.signals import momentum as _lab_momentum


def raw_momentum(
    panel: Panel, f_idx: int, lookback: int = 365, skip: int = 7,
) -> np.ndarray:
    """Raw price momentum via fin-crypto-lab."""
    return _lab_momentum(panel, f_idx, lookback=lookback, skip=skip)


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


def blend_momentum(
    panel: Panel, f_idx: int, lookback: int = 365, skip: int = 7,
    market_col: int | None = None, beta_window: int | None = None,
) -> np.ndarray:
    """50/50 rank-average of raw and residual momentum."""
    raw = raw_momentum(panel, f_idx, lookback=lookback, skip=skip)
    res = residual_momentum(panel, f_idx, lookback=lookback, skip=skip,
                            market_col=market_col, beta_window=beta_window)
    n = len(raw)
    out = np.full(n, np.nan)
    idx = np.where(np.isfinite(raw) & np.isfinite(res))[0]
    if len(idx) < 2:
        return out
    raw_rank = np.empty(len(idx))
    res_rank = np.empty(len(idx))
    raw_rank[np.argsort(raw[idx])] = np.arange(len(idx), dtype=float)
    res_rank[np.argsort(res[idx])] = np.arange(len(idx), dtype=float)
    out[idx] = 0.5 * raw_rank + 0.5 * res_rank
    return out
