import os
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl

# ----------------------------------------------------------------------
# Style (kept consistent with your CIFARNet plotter)
# ----------------------------------------------------------------------
def apply_style(
    *,
    theme="light",
    base_size=16,      # axis labels
    title_size=22,     # subplot title
    suptitle_size=26,  # main title
    legend_size=18,    # legend
    palette="tab10",
    dpi=140, save_dpi=300,
    grid=True
):
    mpl.rcParams.update({
        "axes.titlesize": title_size,
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 1,
        "ytick.labelsize": base_size - 1,
        "legend.fontsize": legend_size,
        "figure.titlesize": suptitle_size,
        "figure.dpi": dpi,
        "savefig.dpi": save_dpi,
        "axes.grid": grid,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelweight": "medium",
        "axes.unicode_minus": False,
    })
    if theme == "dark":
        mpl.rcParams.update({
            "figure.facecolor": "#111112", "axes.facecolor": "#111112",
            "axes.edgecolor": "#BBBBBB", "text.color": "#EAEAEA",
            "axes.labelcolor": "#EAEAEA", "xtick.color": "#EAEAEA",
            "ytick.color": "#EAEAEA", "grid.color": "#444444",
        })
    else:
        mpl.rcParams.update({
            "figure.facecolor": "white", "axes.facecolor": "white",
        })
    if isinstance(palette, str):
        cmap = plt.get_cmap(palette)
        try:
            colors = cmap.colors
        except AttributeError:
            colors = cmap(np.linspace(0, 1, 10))
    else:
        colors = palette
    mpl.rcParams["axes.prop_cycle"] = cycler(color=colors)

def add_figure_legend(fig, axes, *, loc="upper center", ncol=4, title="", bbox_to_anchor=(0.5, 0.985)):
    """Figure-level legend placed above the suptitle by default."""
    handles, labels = [], []
    for ax in np.ravel(axes):
        h, l = ax.get_legend_handles_labels()
        for i, lab in enumerate(l):
            if lab not in labels:
                labels.append(lab)
                handles.append(h[i])
    if handles:
        fig.legend(
            handles, labels,
            loc=loc, bbox_to_anchor=bbox_to_anchor,
            ncol=min(ncol, len(handles)),
            frameon=False, title=title
        )
        for ax in np.ravel(axes):
            leg = ax.get_legend()
            if leg:
                leg.remove()

# ----------------------------------------------------------------------
# Utilities: parse widen -> r, group/aggregate nuclear norm
# ----------------------------------------------------------------------
_R_MAX_WRN = 16 * 3 * 3  # 144 for layer1.0.conv1: in_ch*kh*kw

def _parse_widen_from_name(name: str) -> int | None:
    # Accept "widen=10", "widen = 10", "(widen=10)" etc.
    # Also accept "w=10" just in case.
    m = re.search(r"(?:widen|w)\s*=\s*(\d+)", name)
    return int(m.group(1)) if m else None

def _compute_r_from_widen(w: int) -> int:
    # monitored weight: layer1.0.conv1, shape [16*w, 16, 3, 3]
    # r = min(out_ch, in_ch*kh*kw) = min(16w, 144)
    return int(min(16 * w, _R_MAX_WRN))

def _optimizer_label(name: str) -> str:
    if "SGD with Momentum" in name:
        return "SGD-M"
    if "Muon with SVD" in name:
        return "Muon(SVD)"
    if name.startswith("Muon (steps=") or name.startswith("Muon "):
        return "Muon(NS)"
    return name

def group_rank_scaling_wrn(results: dict, include_svd: bool = True) -> dict[str, list[tuple[int, float, float]]]:
    """
    Returns: { optimizer_label: [(r, mean_over_runs, std_over_runs), ...] }, sorted by r.
    mean_over_runs is computed from per-run epoch-averaged nucnorm.
    """
    groups: dict[str, list[tuple[int, float, float]]] = {}
    for name, res in results.items():
        if "nucnorm" not in res or not res["nucnorm"]:
            continue

        opt = _optimizer_label(name)
        if opt == "Muon(SVD)" and not include_svd:
            continue

        # Prefer recorded r values; fallback to computing from widen in name.
        r_list = res.get("r", [])
        if len(r_list) > 0:
            r_val = int(np.median(r_list))
        else:
            w = _parse_widen_from_name(name)
            r_val = _compute_r_from_widen(w) if w is not None else None
        if r_val is None:
            continue

        runs = res["nucnorm"]  # [runs][epochs]
        run_means = [float(np.mean(epoch_vals)) for epoch_vals in runs if len(epoch_vals) > 0]
        if len(run_means) == 0:
            continue
        mu = float(np.mean(run_means))
        sd = float(np.std(run_means, ddof=1)) if len(run_means) > 1 else 0.0

        groups.setdefault(opt, []).append((r_val, mu, sd))

    for opt in groups:
        groups[opt].sort(key=lambda t: t[0])
    return groups

def fit_loglog_slope(rs, ys):
    """Fit y ~ C * r^beta on log–log; returns (beta, C)."""
    x = np.log(np.asarray(rs, dtype=float))
    y = np.log(np.asarray(ys, dtype=float))
    beta, a = np.polyfit(x, y, 1)  # y = beta*x + a
    C = np.exp(a)
    return float(beta), float(C)

# ----------------------------------------------------------------------
# Main plotting: rank scaling for WRN‑28‑w
# ----------------------------------------------------------------------
def plot_rank_scaling_wrn(
    results: dict,
    out_path: str,
    *,
    include_svd: bool = True,
    theme: str = "light",
    axis_label_size: int = 18,
    legend_size: int = 18,
    linewidth: float = 2.2,
    markersize: int = 7,
    suptitle: str = r"WRN‑28‑$w$ on CIFAR‑10: Nuclear‑norm Rank Scaling",
    right_title: str = r"Normalized by $\sqrt{r}$",
    draw_rmax_line: bool = True
):
    apply_style(theme=theme, base_size=axis_label_size, legend_size=legend_size,
                title_size=22, suptitle_size=26, palette="tab10", dpi=140)

    data = group_rank_scaling_wrn(results, include_svd=include_svd)

    # Show Muon vs SGD-M; include Muon(SVD) optionally
    order = ["SGD-M", "Muon(NS)"] + (["Muon(SVD)"] if include_svd else [])
    order = [opt for opt in order if opt in data]
    if len(order) == 0:
        raise RuntimeError("No WRN rank-scaling data found. Did you run train_wrn_rank.py and log 'nucnorm' and 'r'?")

    # Consistent colors/markers
    base_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    base_markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    style_map = {opt: (base_colors[i % len(base_colors)], base_markers[i % len(base_markers)])
                 for i, opt in enumerate(order)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax_raw, ax_norm = axes

    # Left: raw scaling (log–log)
    for opt in order:
        tup = data[opt]
        rs = [r for (r, mu, sd) in tup]
        ys = [mu for (r, mu, sd) in tup]
        es = [sd for (r, mu, sd) in tup]
        c, m = style_map[opt]
        ax_raw.errorbar(rs, ys, yerr=es, fmt=m, color=c, ecolor=c,
                        elinewidth=1.0, lw=0, ms=markersize, capsize=3, label=opt)
        ax_raw.plot(rs, ys, "-", lw=linewidth, color=c)
        if len(rs) >= 2 and all(y > 0 for y in ys):
            beta, _ = fit_loglog_slope(rs, ys)
            ax_raw.text(rs[-1], ys[-1], f" β≈{beta:.2f}",
                        fontsize=axis_label_size-2, ha="left", va="bottom", color=c)
            print(f"[WRN log–log slope] {opt}: β ≈ {beta:.3f}")

    ax_raw.set_xscale("log"); ax_raw.set_yscale("log")
    ax_raw.set_xlabel(r"$r=\min\{m,n\}$")
    ax_raw.set_ylabel(r"avg. $\|\nabla f(W)\|_*$ per epoch")
    ax_raw.set_title("Raw scaling (log–log)")

    if draw_rmax_line:
        ax_raw.axvline(_R_MAX_WRN, linestyle=":", linewidth=1.2, color="gray", alpha=0.85)
        ymin, ymax = ax_raw.get_ylim()
        ax_raw.text(_R_MAX_WRN, ymin, " r_max = 144 ", rotation=90,
                    va="bottom", ha="right", fontsize=axis_label_size-3, color="gray")

    # Right: normalized by sqrt(r)
    for opt in order:
        tup = data[opt]
        rs = np.array([r for (r, mu, sd) in tup], dtype=float)
        ys = np.array([mu for (r, mu, sd) in tup], dtype=float)
        es = np.array([sd for (r, mu, sd) in tup], dtype=float)
        norm = np.sqrt(rs)
        c, m = style_map[opt]
        y_div = ys / norm
        e_div = es / norm
        ax_norm.errorbar(rs, y_div, yerr=e_div, fmt=m, color=c, ecolor=c,
                         elinewidth=1.0, lw=0, ms=markersize, capsize=3, label=opt)
        ax_norm.plot(rs, y_div, "-", lw=linewidth, color=c)
        if len(rs) >= 2 and all(y > 0 for y in y_div):
            beta_norm, _ = fit_loglog_slope(rs, y_div)
            ax_norm.text(rs[-1], y_div[-1], f" β≈{beta_norm:.2f}",
                         fontsize=axis_label_size-2, ha="left", va="bottom", color=c)
            print(f"[WRN log–log slope (normalized)] {opt}: β ≈ {beta_norm:.3f}")

    ax_norm.set_xscale("log")
    ax_norm.set_xlabel(r"$r=\min\{m,n\}$")
    ax_norm.set_ylabel(r"avg. $\|\nabla f(W)\|_*\,/\,\sqrt{r}$ per epoch")
    ax_norm.set_title(right_title)

    # Cosmetics
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=False, min_n_ticks=3))
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

    fig.suptitle(suptitle, y=0.985)
    add_figure_legend(fig, np.array(axes), loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.955))
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.90])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"📈 WRN rank-scaling plot saved → {os.path.abspath(out_path)}")

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot WRN‑28‑w rank scaling from train_wrn_rank.py results")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to results.json produced by train_wrn_rank.py")
    parser.add_argument("--outfile", type=str, default=None,
                        help="Output PDF/PNG path. Default: <results_dir>/wrn_rank_scaling.pdf")
    parser.add_argument("--include_svd", action="store_true", default=True,
                        help="Also show Muon(SVD) curves.")
    parser.add_argument("--theme", type=str, default="light", choices=["light", "dark"])
    args = parser.parse_args()

    with open(args.results, "r") as f:
        results = json.load(f)

    out_path = args.outfile
    if out_path is None:
        out_dir = os.path.dirname(os.path.abspath(args.results))
        out_path = os.path.join(out_dir, "wrn_rank_scaling.pdf")

    plot_rank_scaling_wrn(
        results,
        out_path,
        include_svd=args.include_svd,
        theme=args.theme,
        suptitle=r"WRN‑28‑$w$ on CIFAR‑10: Nuclear‑norm Rank Scaling",
        right_title=r"Normalized by $\sqrt{r}$ (monitor: layer1.0.conv1)",
        draw_rmax_line=True,
    )

if __name__ == "__main__":
    main()
