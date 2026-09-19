"""Residual momentum sweep grid. Extends fin-crypto-lab's grid with
multiple lookbacks and beta estimation windows."""

GRID = [
    {"instrument": inst, "top_n": n, "lookback": lb, "skip": 7, "beta_window": bw}
    for inst in ("spot", "futures")
    for lb in (90, 180, 365)
    for n in (10, 20, 30)
    for bw in (None,)  # Phase 3 adds (60, 90, 120)
]


def config_name(g: dict) -> str:
    bw = g.get("beta_window")
    bw_tag = f"_bw{bw}" if bw is not None else ""
    return f"{g['instrument']}_resmom_lb{g['lookback']}_top{g['top_n']}{bw_tag}"
