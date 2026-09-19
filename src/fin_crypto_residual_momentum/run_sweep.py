"""Momentum sweep runner with full honesty battery.
Run: uv run python -m fin_crypto_residual_momentum --instrument spot --signal residual
     uv run python -m fin_crypto_residual_momentum --instrument spot --signal raw
     uv run python -m fin_crypto_residual_momentum --instrument spot --signal blend
     uv run python -m fin_crypto_residual_momentum --instrument spot --signal compare
Exit 0 = PASS, 1 = FAIL, 2 = kill condition."""
import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl

from fin_crypto_lab import config
from fin_crypto_lab.backtest import slippage_sweep
from fin_crypto_lab.metrics_overfit import (
    deflated_sharpe_ratio,
    pbo_cscv,
    sharpe_moments,
)
from fin_crypto_lab.panel import build_panel, crypto_sessions, weekly_formations
from fin_crypto_lab.sweep import (
    THRESHOLDS,
    benchmark_targets,
    period_returns,
    topn_targets,
    weekly_cagr,
    weekly_sharpe,
)
from fin_crypto_lab.universe import build_universe

from fin_crypto_residual_momentum.signals import (
    blend_momentum,
    raw_momentum,
    residual_momentum,
)
from fin_crypto_residual_momentum.sweep import (
    BLEND_GRID,
    RAWMOM_GRID,
    RESMOM_GRID,
    config_name,
)

log = logging.getLogger("fin_crypto_residual_momentum.sweep")

RESULTS_DIR = Path("results")


def write_verdict(rows, checks, family_pass, extra, run_label) -> Path:
    out_dir = RESULTS_DIR / run_label
    out_dir.mkdir(parents=True, exist_ok=True)
    sig_label = extra.get("signal_type", "resmom")
    lines = [
        f"# {sig_label} momentum {extra.get('instrument', '')} verdict "
        f"({dt.date.today()})",
        "",
        f"## FAMILY: {'PASS' if family_pass else 'FAIL'}",
        "",
        f"Selected: **{extra['selected']}** "
        f"(DSR {extra['dsr']:.3f}, PBO {extra['pbo']:.3f})",
        "",
        "| check | result | measured |",
        "|---|---|---|",
    ]
    for cid, passed, meas in checks:
        lines.append(f"| {cid} | {'PASS' if passed else 'FAIL'} | {meas} |")
    lines += [
        "",
        "## Grid",
        "",
        "| config | train Sharpe | test Sharpe | test CAGR | full CAGR "
        "| turnover | min names |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: -x["train_sharpe"]):
        label = f"**{r['name']}** **selected**" if r["selected"] else r["name"]
        lines.append(
            f"| {label} | {r['train_sharpe']:.2f} | {r['test_sharpe']:.2f} "
            f"| {r['test_cagr']:.2%} | {r['full_cagr']:.2%} "
            f"| {r['turnover']:.2f} | {r['min_names']} |"
        )
    (out_dir / "verdict.md").write_text("\n".join(lines) + "\n")
    return out_dir


def write_comparison(all_rows: dict[str, list[dict]], instrument,
                     bm_test_sharpe) -> Path:
    """Side-by-side comparison: all signal types."""
    out_dir = RESULTS_DIR / f"crypto_{instrument}_compare_{dt.date.today().isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_key: dict[tuple, dict[str, dict]] = {}
    for sig_type, rows in all_rows.items():
        for r in rows:
            by_key.setdefault((r["lookback"], r["top_n"]), {})[sig_type] = r

    signal_types = list(all_rows.keys())
    header_cols = []
    for st in signal_types:
        header_cols += [f"{st} train SR", f"{st} test SR"]
    header = "| lookback | top_n | " + " | ".join(header_cols) + " |"
    sep = "|---|---" + "|---" * len(header_cols) + "|"

    lines = [
        f"# Momentum comparison: {instrument} ({dt.date.today()})",
        "",
        f"Benchmark test Sharpe: {bm_test_sharpe:.2f}",
        "",
        header,
        sep,
    ]
    for (lb, n) in sorted(by_key):
        cells = []
        for st in signal_types:
            r = by_key[(lb, n)].get(st)
            if r:
                cells += [f"{r['train_sharpe']:.2f}", f"{r['test_sharpe']:.2f}"]
            else:
                cells += ["", ""]
        lines.append(f"| {lb} | {n} | " + " | ".join(cells) + " |")

    lines += ["", "## Selected configs"]
    for st in signal_types:
        sel = next((r for r in all_rows[st] if r["selected"]), None)
        if sel:
            lines.append(
                f"- {st.capitalize()}: **{sel['name']}** "
                f"(train SR {sel['train_sharpe']:.2f}, "
                f"test SR {sel['test_sharpe']:.2f})"
            )

    lines += ["", "## Pairwise summary"]
    pairs = [("residual", "raw"), ("blend", "raw"), ("blend", "residual")]
    for a, b in pairs:
        if a not in all_rows or b not in all_rows:
            continue
        deltas = [
            by_key[k][a]["test_sharpe"] - by_key[k][b]["test_sharpe"]
            for k in by_key if a in by_key[k] and b in by_key[k]
        ]
        if not deltas:
            continue
        avg = np.mean(deltas)
        wins = sum(1 for d in deltas if d > 0)
        lines.append(
            f"- {a.capitalize()} vs. {b}: mean delta {avg:+.3f}, "
            f"wins {wins}/{len(deltas)}"
        )

    best_st = max(signal_types,
                  key=lambda st: np.mean([r["test_sharpe"] for r in all_rows[st]]))
    lines += [
        "",
        f"Best average test SR: **{best_st}** "
        f"({np.mean([r['test_sharpe'] for r in all_rows[best_st]]):.3f})",
    ]

    path = out_dir / "comparison.md"
    path.write_text("\n".join(lines) + "\n")
    return out_dir


def _setup_instrument(instrument: str):
    """Return (data_dir, cost_fn, slip_levels, decision_slip, form_start,
    form_end, train_end) for the given instrument."""
    data_dir = (config.SPOT_DATA_DIR if instrument == "spot"
                else config.FUTURES_DATA_DIR)
    cost_fn = (config.spot_cost_frac if instrument == "spot"
               else config.futures_cost_frac)
    slip_levels = (config.SPOT_SLIP_LEVELS_BP if instrument == "spot"
                   else config.FUTURES_SLIP_LEVELS_BP)
    decision_slip = (config.SPOT_DECISION_SLIP_BP if instrument == "spot"
                     else config.FUTURES_DECISION_SLIP_BP)
    if instrument == "futures":
        form_start = config.FUTURES_FORM_START
        form_end = config.FUTURES_FORM_END
        train_end = config.FUTURES_TRAIN_END
    else:
        form_start = config.FORM_START
        form_end = config.FORM_END
        train_end = config.TRAIN_END
    return data_dir, cost_fn, slip_levels, decision_slip, form_start, form_end, train_end


def _build_common(instrument):
    """Build shared objects: sessions, formations, universe, panel, f_idx, market_col."""
    (data_dir, cost_fn, slip_levels, decision_slip,
     form_start, form_end, train_end) = _setup_instrument(instrument)

    sessions = crypto_sessions(
        str(form_start - dt.timedelta(days=400)), str(form_end))
    formations = [d for d in weekly_formations(sessions)
                  if form_start <= d <= form_end]

    ohlcv_dir = data_dir / "ohlcv"
    max_n = 30
    universe = build_universe(ohlcv_dir, formations, lookback=90, top_n=max_n)
    log.info("universe: %d snapshots, %d unique symbols",
             universe["snapshot_date"].n_unique(),
             universe["symbol"].n_unique())

    symbols = sorted(universe["symbol"].unique().to_list())
    panel = build_panel(symbols, sessions, data_dir=data_dir)
    f_idx = {d: panel.session_index(d) for d in formations}

    market_col = None
    for mkt_sym in ("XBTUSD", "BTCUSD"):
        if mkt_sym in symbols:
            market_col = symbols.index(mkt_sym)
            break
    log.info("market_col=%s (%s)",
             market_col, symbols[market_col] if market_col is not None else "EW")

    bm_targets = benchmark_targets(universe, formations)
    bm_sw = slippage_sweep(panel, bm_targets, nav0=config.NAV_DEFAULT,
                           slip_levels=slip_levels, cost_frac_fn=cost_fn)
    bm_res = bm_sw[decision_slip]
    bm_weekly = period_returns(bm_res.nav, formations)
    bm_marks = bm_res.nav.filter(
        pl.col("date").is_in(formations)).sort("date")
    bm_dates = bm_marks["date"].to_list()[1:]
    bm_test = [r for r, d in zip(bm_weekly, bm_dates) if d > train_end]
    bm_test_sharpe = weekly_sharpe(bm_test)

    return (panel, formations, f_idx, universe, market_col,
            cost_fn, slip_levels, decision_slip, train_end,
            bm_test_sharpe)


def run_signal(instrument: str, signal_type: str,
               common=None) -> tuple[list[dict], int, float]:
    """Run one signal type through the sweep and honesty battery.
    Returns (rows, exit_code, bm_test_sharpe)."""
    if common is None:
        common = _build_common(instrument)
    (panel, formations, f_idx, universe, market_col,
     cost_fn, slip_levels, decision_slip, train_end,
     bm_test_sharpe) = common

    grids = {"residual": RESMOM_GRID, "raw": RAWMOM_GRID, "blend": BLEND_GRID}
    grid = grids[signal_type]
    inst_grid = [g for g in grid if g["instrument"] == instrument]

    sig_cache: dict[tuple, dict[dt.date, dict[str, float]]] = {}

    def get_signals(lb: int, sk: int, bw: int | None = None):
        key = (signal_type, lb, sk, bw)
        if key in sig_cache:
            return sig_cache[key]
        sig_by_form: dict[dt.date, dict[str, float]] = {}
        for d in formations:
            fi = f_idx[d]
            if fi < lb:
                continue
            if signal_type == "residual":
                raw = residual_momentum(panel, fi, lookback=lb, skip=sk,
                                        market_col=market_col, beta_window=bw)
            elif signal_type == "blend":
                raw = blend_momentum(panel, fi, lookback=lb, skip=sk,
                                     market_col=market_col, beta_window=bw)
            else:
                raw = raw_momentum(panel, fi, lookback=lb, skip=sk)
            sig_by_form[d] = dict(zip(panel.symbols, raw.tolist()))
        sig_cache[key] = sig_by_form
        return sig_by_form

    rows, weekly_own = [], {}
    results_cache: dict[str, dict] = {}
    nonfinite = False

    for g in inst_grid:
        name = config_name(g, signal_type)
        sig_by_form = get_signals(g["lookback"], g["skip"], g.get("beta_window"))
        tgt = topn_targets(sig_by_form, universe, n=g["top_n"],
                           min_names=THRESHOLDS["KC6_MIN_NAMES"])
        sw = slippage_sweep(panel, tgt, nav0=config.NAV_DEFAULT,
                            slip_levels=slip_levels, cost_frac_fn=cost_fn)
        res = sw[decision_slip]
        results_cache[name] = {"sw": sw, "res": res, "tgt": tgt}
        nav = res.nav["nav"].to_numpy()
        nonfinite |= bool((~np.isfinite(nav)).any() or (nav <= 0).any())

        cfg_weekly = period_returns(res.nav, formations)
        weekly_own[name] = list(cfg_weekly)

        marks = res.nav.filter(
            pl.col("date").is_in(formations)).sort("date")
        dates = marks["date"].to_list()[1:]
        train = [r for r, d in zip(cfg_weekly, dates) if d <= train_end]
        test = [r for r, d in zip(cfg_weekly, dates) if d > train_end]
        n_days = res.nav.height - 1

        names_per_form = (
            tgt.group_by("formation_date")
            .agg(pl.col("symbol").n_unique().alias("n_names"))
        )
        min_names = (int(names_per_form["n_names"].min())
                     if names_per_form.height else 0)

        rows.append({
            "name": name,
            "lookback": g["lookback"],
            "top_n": g["top_n"],
            "train_sharpe": weekly_sharpe(train),
            "test_sharpe": weekly_sharpe(test),
            "test_cagr": weekly_cagr(test),
            "full_cagr": weekly_cagr(cfg_weekly),
            "turnover": float(
                res.turnover["turnover"].sum() * 365 / n_days),
            "min_names": min_names,
            "selected": False,
        })
        log.info("%s: train SR %.2f test SR %.2f", name,
                 rows[-1]["train_sharpe"], rows[-1]["test_sharpe"])

    sel = max(rows, key=lambda r: r["train_sharpe"])
    sel["selected"] = True

    own_sel = np.asarray(weekly_own[sel["name"]])
    sr, skew, kurt = sharpe_moments(own_sel)
    trial_srs = [
        float(np.mean(np.asarray(weekly_own[r["name"]])) /
              np.std(np.asarray(weekly_own[r["name"]]), ddof=1))
        for r in rows
    ]
    dsr = deflated_sharpe_ratio(sr_hat=sr, t_obs=len(own_sel), skew=skew,
                                kurt=kurt, trial_sharpes=trial_srs)

    matrix = np.column_stack([weekly_own[r["name"]] for r in rows])
    pbo = pbo_cscv(matrix, s_blocks=THRESHOLDS["S_BLOCKS"])

    cached = results_cache[sel["name"]]
    sel_res = cached["res"]
    sel_sw = cached["sw"]
    sel_tgt = cached["tgt"]

    sel_nav_test = sel_res.nav.filter(pl.col("date") > train_end)
    nav_s = sel_nav_test["nav"]
    test_dd = float(-(nav_s / nav_s.cum_max() - 1.0).min())

    gross_nav = sel_sw[0.0].nav
    n_days_sel = sel_res.nav.height - 1
    years_sel = n_days_sel / 365.0
    drag = (
        (gross_nav["nav"][-1] / gross_nav["nav"][0]) ** (1 / years_sel)
        - (sel_res.nav["nav"][-1] / sel_res.nav["nav"][0]) ** (1 / years_sel)
    )
    costs_frac = (float(sel_res.trades["cost"].sum())
                  / float(sel_res.nav["nav"].mean()) / years_sel)
    recon_err = (abs(drag - costs_frac) / costs_frac
                 if costs_frac else float("inf"))

    weight_sums = (
        sel_tgt.group_by("formation_date")
        .agg(pl.col("weight").sum().alias("wsum"))
    )
    min_wsum = float(weight_sums["wsum"].min())
    max_wsum = float(weight_sums["wsum"].max())
    tol = THRESHOLDS["KC5_WEIGHT_TOL"]
    kc5_pass = (min_wsum >= 1.0 - tol) and (max_wsum <= 1.0 + tol)

    sel_min_names = sel["min_names"]

    checks = [
        ("PC-1", sel["test_sharpe"] >= bm_test_sharpe,
         f"{sel['test_sharpe']:.2f} vs benchmark {bm_test_sharpe:.2f}"),
        ("PC-3", dsr >= THRESHOLDS["DSR_MIN"], f"DSR {dsr:.3f}"),
        ("PC-4", pbo <= THRESHOLDS["PBO_MAX"], f"PBO {pbo:.3f}"),
        ("PC-5", recon_err <= THRESHOLDS["COST_RECON_TOL"],
         f"drag {drag:.2%} vs costs {costs_frac:.2%} "
         f"(err {recon_err:.0%})"),
        ("KC-1",
         all(r["full_cagr"] <= THRESHOLDS["KC1_MAX_CAGR"] for r in rows),
         f"max full CAGR {max(r['full_cagr'] for r in rows):.2%}"),
        ("KC-2",
         all(r["turnover"] <= THRESHOLDS["KC2_MAX_TURNOVER"]
             for r in rows),
         f"max turnover {max(r['turnover'] for r in rows):.2f}"),
        ("KC-3", not nonfinite, str(not nonfinite)),
        ("KC-4", test_dd <= THRESHOLDS["KC4_MAX_DD"],
         f"test maxDD {test_dd:.2%}"),
        ("KC-5", kc5_pass,
         f"min Σw {min_wsum:.4f}, max {max_wsum:.4f}"),
        ("KC-6", sel_min_names >= THRESHOLDS["KC6_MIN_NAMES"],
         f"min names {sel_min_names}"),
    ]
    kills = [c for c in checks if c[0].startswith("KC") and not c[1]]
    family_pass = all(passed for _, passed, _ in checks)

    from fin_crypto_residual_momentum.sweep import SIGNAL_TAGS
    tag = SIGNAL_TAGS[signal_type]
    label = f"crypto_{instrument}_{tag}_{dt.date.today().isoformat()}"
    out = write_verdict(
        rows, checks, family_pass,
        {"selected": sel["name"], "dsr": dsr, "pbo": pbo,
         "instrument": instrument, "signal_type": signal_type},
        run_label=label,
    )
    for cid, passed, meas in checks:
        log.info("%-5s %s  %s", cid, "PASS" if passed else "FAIL", meas)
    log.info("FAMILY: %s — %s", "PASS" if family_pass else "FAIL", out)

    exit_code = 2 if kills else (0 if family_pass else 1)
    return rows, exit_code, bm_test_sharpe


def run_compare(instrument: str) -> int:
    """Run all three signals, then produce a head-to-head comparison."""
    common = _build_common(instrument)
    all_rows = {}
    worst_exit = 0
    for sig in ("residual", "raw", "blend"):
        log.info("=== Running %s momentum ===", sig.upper())
        rows, exit_code, bm_test_sharpe = run_signal(instrument, sig, common)
        all_rows[sig] = rows
        worst_exit = max(worst_exit, exit_code)

    out = write_comparison(all_rows, instrument, bm_test_sharpe)
    log.info("Comparison written to %s", out)

    return worst_exit


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", required=True,
                        choices=["spot", "futures"])
    parser.add_argument("--signal", default="residual",
                        choices=["residual", "raw", "blend", "compare"])
    args = parser.parse_args()

    if args.signal == "compare":
        return run_compare(args.instrument)
    _, exit_code, _ = run_signal(args.instrument, args.signal)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
