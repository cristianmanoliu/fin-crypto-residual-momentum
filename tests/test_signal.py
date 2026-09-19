"""Smoke test: momentum signals produce finite values and
beta_window parameter changes the residual output."""
import datetime as dt

import numpy as np
import polars as pl

from fin_crypto_residual_momentum.signals import raw_momentum, residual_momentum


def _make_panel(n_days=200, n_syms=5, seed=42):
    """Synthetic panel with known BTC-correlated returns."""
    from fin_crypto_lab.panel import Panel

    rng = np.random.default_rng(seed)
    start = dt.date(2020, 1, 1)
    sessions = pl.date_range(start, start + dt.timedelta(days=n_days - 1), eager=True)
    btc_rets = rng.normal(0.001, 0.03, n_days)
    prices = np.zeros((n_days, n_syms))
    prices[0] = 100.0
    for t in range(1, n_days):
        for j in range(n_syms):
            beta = 0.5 + j * 0.3
            idio = rng.normal(0.0005 * (j - 2), 0.01)
            prices[t, j] = prices[t - 1, j] * (1 + beta * btc_rets[t] + idio)

    symbols = [f"SYM{i}" for i in range(n_syms)]
    return Panel(
        sessions=sessions,
        symbols=symbols,
        open=prices.copy(),
        high=prices * 1.01,
        low=prices * 0.99,
        close=prices.copy(),
        close_ff=prices.copy(),
        volume=np.ones_like(prices) * 1000,
        vwap=prices.copy(),
        tradable=np.ones_like(prices, dtype=bool),
    )


def test_residual_momentum_basic():
    panel = _make_panel()
    sig = residual_momentum(panel, f_idx=180, lookback=90, skip=7)
    finite = np.isfinite(sig)
    assert finite.sum() >= 3, f"too few finite signals: {finite.sum()}"


def test_raw_momentum_basic():
    panel = _make_panel()
    sig = raw_momentum(panel, f_idx=180, lookback=90, skip=7)
    finite = np.isfinite(sig)
    assert finite.sum() >= 3, f"too few finite signals: {finite.sum()}"


def test_raw_vs_residual_differ():
    panel = _make_panel()
    raw = raw_momentum(panel, f_idx=180, lookback=90, skip=7)
    res = residual_momentum(panel, f_idx=180, lookback=90, skip=7)
    mask = np.isfinite(raw) & np.isfinite(res)
    assert not np.allclose(raw[mask], res[mask]), \
        "raw and residual should produce different signals"


def test_beta_window_changes_output():
    panel = _make_panel()
    sig_full = residual_momentum(panel, f_idx=180, lookback=90, skip=7,
                                 beta_window=None)
    sig_short = residual_momentum(panel, f_idx=180, lookback=90, skip=7,
                                  beta_window=60)
    assert not np.allclose(sig_full[np.isfinite(sig_full)],
                           sig_short[np.isfinite(sig_short)]), \
        "beta_window should change the signal"
