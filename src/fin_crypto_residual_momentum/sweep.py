"""Sweep grids for residual, raw, and blend momentum signals."""

RESMOM_GRID = [
    {"instrument": inst, "top_n": n, "lookback": lb, "skip": 7, "beta_window": bw}
    for inst in ("spot", "futures")
    for lb in (90, 180, 365)
    for n in (10, 20, 30)
    for bw in (None,)  # Phase 3 adds (60, 90, 120)
]

RAWMOM_GRID = [
    {"instrument": inst, "top_n": n, "lookback": lb, "skip": 7}
    for inst in ("spot", "futures")
    for lb in (90, 180, 365)
    for n in (10, 20, 30)
]

BLEND_GRID = [
    {"instrument": inst, "top_n": n, "lookback": lb, "skip": 7, "beta_window": bw}
    for inst in ("spot", "futures")
    for lb in (90, 180, 365)
    for n in (10, 20, 30)
    for bw in (None,)
]

GRID = RESMOM_GRID  # backwards compat

SIGNAL_TAGS = {"residual": "resmom", "raw": "rawmom", "blend": "blend"}


def config_name(g: dict, signal_type: str = "resmom") -> str:
    bw = g.get("beta_window")
    bw_tag = f"_bw{bw}" if bw is not None else ""
    tag = SIGNAL_TAGS.get(signal_type, signal_type)
    return f"{g['instrument']}_{tag}_lb{g['lookback']}_top{g['top_n']}{bw_tag}"
