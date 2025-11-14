import os
import json
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.ticker import MaxNLocator, FuncFormatter
import matplotlib as mpl

# ----------------------------- Styling ---------------------------------
def apply_style(
    *,
    theme="light",
    base_size=16,
    title_size=24,
    suptitle_size=26,
    legend_size=18,
    palette="tab10",
    dpi=140, save_dpi=300,
    grid=True
):
    """Applies a global Matplotlib style."""
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
    mpl.rcParams["axes.unicode_minus"] = False

def _format_seconds(x, _pos):
    return f"{int(x)}s"

def add_figure_legend(fig, axes, *, loc="upper center", ncol=6, title="Optimizers"):
    """Adds a single, unified legend for the entire figure."""
    handles, labels = [], []
    for ax in axes.ravel():
        h, l = ax.get_legend_handles_labels()
        for i, label in enumerate(l):
            if label not in labels:
                labels.append(label)
                handles.append(h[i])
    if handles:
        fig.legend(
            handles, labels,
            loc=loc, bbox_to_anchor=(0.5, 0.96),
            ncol=min(ncol, len(handles)),
            frameon=False, title=title
        )
        for ax in axes.ravel():
            if ax.get_legend():
                ax.get_legend().remove()

# ----------------------------- Parsers & limits -------------------------
def _parse_limit_token(tok: str | None) -> float | None | str:
    """
    Accept tokens:
      - 'min', 'max', 'auto', 'none'
      - numeric (e.g., 1.0)
      - percentage for accuracy (e.g., '80%')
      - seconds for time x-lims (e.g., '20s')
    Returns float | None | {'min','max','auto','none'}.
    """
    if tok is None:
        return None
    s = tok.strip().lower()
    if s in {"min", "max", "auto", "none"}:
        return s
    if s.endswith("%"):  # accuracy percents
        try:
            return float(s[:-1]) / 100.0
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid percentage: {tok}")
    if s.endswith("s"):  # time in seconds
        try:
            return float(s[:-1])
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid seconds token: {tok}")
    try:
        return float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid limit token: {tok}")

def _parse_lims(s: str | None):
    """Parse '<low>,<high>' strings into tokens understood by _resolve_limits."""
    if s is None:
        return (None, None)
    s = s.strip().lower()
    if s in {"none", "auto"}:
        return (s, s)
    if "," not in s:
        raise argparse.ArgumentTypeError("Expected '<low>,<high>' (e.g., '12,max').")
    lo, hi = s.split(",", 1)
    return (_parse_limit_token(lo), _parse_limit_token(hi))

def _resolve_limits(lo_tok, hi_tok, data_min, data_max, *, default=None, clip=(None, None), pad=0.03):
    """
    Convert tokens to numeric (lo, hi). 'min'/'max' bind to data extents.
    'none'/'auto'/None -> default (no change). Optional clipping and padding.
    """
    if default is None:
        default = (None, None)

    def _tok_to_val(tok):
        if tok in {None, "none", "auto"}:
            return None
        if tok == "min":
            return data_min
        if tok == "max":
            return data_max
        return tok  # numeric

    lo = _tok_to_val(lo_tok)
    hi = _tok_to_val(hi_tok)

    if lo is None and hi is None:
        return default
    if lo is None:
        lo = data_min
    if hi is None:
        hi = data_max

    lo_clip, hi_clip = clip
    if lo_clip is not None:
        lo = max(lo, lo_clip)
    if hi_clip is not None:
        hi = min(hi, hi_clip)

    if not (lo < hi):
        eps = 1e-6
        lo = min(lo, hi - eps)
        hi = max(hi, lo + eps)

    span = max(1e-12, hi - lo)
    if lo_tok not in {"min"}:
        lo = lo + pad * span
    if hi_tok not in {"max"}:
        hi = hi - pad * span
    return (lo, hi)

# ----------------------------- Plotting --------------------------------
def plot_comparison(
    results: dict, 
    out_path: str,
    *,
    suptitle="",
    axis_label_size=16,
    legend_size=18,
    acc_as_percent=True,
    linewidth=2.2,
    markersize=6,
    fill_alpha=0.2,
    legend_loc="upper center",
    legend_ncol=6,
    # y-axis zoom (applies to both loss panels / both acc panels)
    ylims_loss=("none", "none"),
    ylims_acc=("none", "none"),
    # x-axis zoom per panel
    xlims_epoch_loss=("none", "none"),
    xlims_epoch_acc=("none", "none"),
    xlims_time_loss=("none", "none"),
    xlims_time_acc=("none", "none"),
):
    """
    Visualize and compare multiple experiment results in a 2x2 grid,
    with independent x/y zoom controls per panel.
    """
    # 1. Apply style
    apply_style(
        theme="light",
        base_size=axis_label_size,
        legend_size=legend_size, 
        title_size=22,
        suptitle_size=26,
        palette="tab10",
        dpi=140,
    )
    
    COLORS = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    MARKERS = itertools.cycle(["o", "s", "^", "d", "v", "P", "X", "*"])

    # 2. Create Figure and Axes (no sharex to allow independent cropping)
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    if suptitle:
        fig.suptitle(suptitle, y=0.98) 
    
    ax_epoch_loss, ax_epoch_acc = axes[0, 0], axes[0, 1]
    ax_time_loss,  ax_time_acc  = axes[1, 0], axes[1, 1]

    # Track global ranges for autoscaling/limits
    _loss_low, _loss_high = [], []
    _acc_low,  _acc_high  = [], []
    _epoch_min, _epoch_max = [], []
    _time_min,  _time_max  = [], []

    # 3. Iterate through data and plot
    for name, res in results.items():
        if not res.get("train"):
            continue
            
        # Ensure data lengths are consistent
        min_len = min(len(r) for r in res.get("train", []))
        tr = np.stack([np.asarray(r)[:min_len] for r in res["train"]])
        acc = np.stack([np.asarray(r)[:min_len] for r in res["acc"]])
        t = np.stack([np.asarray(r)[:min_len] for r in res["time"]])

        # Calculate mean and standard deviation
        mu_t, mu_tr, mu_acc = t.mean(0), tr.mean(0), acc.mean(0)
        std_tr, std_acc = tr.std(0), acc.std(0)
        epochs = np.arange(1, min_len + 1)

        # Accumulate ranges (include ± std envelope)
        _loss_low.append(np.min(mu_tr - std_tr)); _loss_high.append(np.max(mu_tr + std_tr))
        _acc_low.append(np.min(mu_acc - std_acc)); _acc_high.append(np.max(mu_acc + std_acc))
        _epoch_min.append(float(epochs.min()));    _epoch_max.append(float(epochs.max()))
        _time_min.append(float(mu_t.min()));       _time_max.append(float(mu_t.max()))
        
        c, m = next(COLORS), next(MARKERS)

        # Epoch-wise plots
        markevery = max(1, min_len // 10)
        ax_epoch_loss.plot(epochs, mu_tr, marker=m, linestyle="-", c=c, label=name,
                           linewidth=linewidth, markersize=markersize, markevery=markevery)
        ax_epoch_loss.fill_between(epochs, mu_tr - std_tr, mu_tr + std_tr, alpha=fill_alpha, color=c)
        
        ax_epoch_acc.plot(epochs, mu_acc, marker=m, linestyle="-", c=c, label=name,
                          linewidth=linewidth, markersize=markersize, markevery=markevery)
        ax_epoch_acc.fill_between(epochs, mu_acc - std_acc, mu_acc + std_acc, alpha=fill_alpha, color=c)

        # Time-wise plots
        ax_time_loss.plot(mu_t, mu_tr, marker=m, linestyle="-", c=c, label=name,
                          linewidth=linewidth, markersize=markersize, markevery=markevery)
        ax_time_loss.fill_between(mu_t, mu_tr - std_tr, mu_tr + std_tr, alpha=fill_alpha, color=c)
        
        ax_time_acc.plot(mu_t, mu_acc, marker=m, linestyle="-", c=c, label=name,
                         linewidth=linewidth, markersize=markersize, markevery=markevery)
        ax_time_acc.fill_between(mu_t, mu_acc - std_acc, mu_acc + std_acc, alpha=fill_alpha, color=c)

    # 4. Set titles and labels
    ax_epoch_loss.set(title="Train Loss vs. Epoch", xlabel="Epoch", ylabel="Loss")
    ax_epoch_acc.set(title="Test / Val Accuracy vs. Epoch", xlabel="Epoch", ylabel="Accuracy(%)")
    ax_time_loss.set(title="Train Loss vs. Wall-Clock Time", xlabel="Wall-Clock Time (s)", ylabel="Loss")
    ax_time_acc.set(title="Test / Val Accuracy vs. Wall-Clock Time", xlabel="Wall-Clock Time (s)", ylabel="Accuracy(%)") 

    # 5. Configure axis formats and ticks (no shared x to allow independent x-lims)
    ax_epoch_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_epoch_acc.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_time_loss.xaxis.set_major_formatter(FuncFormatter(_format_seconds))
    ax_time_acc.xaxis.set_major_formatter(FuncFormatter(_format_seconds))

    if acc_as_percent:
        formatter = FuncFormatter(lambda x, _pos: f'{int(round(x * 100))}')
        ax_epoch_acc.yaxis.set_major_formatter(formatter)
        ax_time_acc.yaxis.set_major_formatter(formatter)

    for ax in axes.ravel():
        ax.minorticks_on()
        ax.tick_params(axis="both", which="major", length=6, width=1.2)
        ax.tick_params(axis="both", which="minor", length=3, width=0.6)
        ax.grid(True, which="major", linestyle="--", alpha=0.6)
        ax.grid(True, which="minor", linestyle=":", alpha=0.3)

    # 6. Add global legend
    add_figure_legend(fig, axes, loc=legend_loc, ncol=legend_ncol, title="")

    # 7. Resolve & apply y-limits (zoom)
    data_loss_min, data_loss_max = float(np.min(_loss_low)),  float(np.max(_loss_high))
    data_acc_min,  data_acc_max  = float(np.min(_acc_low)),   float(np.max(_acc_high))

    loss_lo, loss_hi = _resolve_limits(
        ylims_loss[0], ylims_loss[1], data_loss_min, data_loss_max,
        default=(None, None), clip=(None, None), pad=0.03
    )
    if loss_lo is not None or loss_hi is not None:
        ax_epoch_loss.set_ylim(loss_lo, loss_hi)
        ax_time_loss.set_ylim(loss_lo, loss_hi)

    acc_lo, acc_hi = _resolve_limits(
        ylims_acc[0], ylims_acc[1], data_acc_min, data_acc_max,
        default=(None, None), clip=(0.0, 1.0), pad=0.03
    )
    if acc_lo is not None or acc_hi is not None:
        ax_epoch_acc.set_ylim(acc_lo, acc_hi)
        ax_time_acc.set_ylim(acc_lo, acc_hi)

    # 8. Resolve & apply x-limits (per panel)
    data_epoch_min, data_epoch_max = float(np.min(_epoch_min)), float(np.max(_epoch_max))
    data_time_min,  data_time_max  = float(np.min(_time_min)),  float(np.max(_time_max))

    el_lo, el_hi = _resolve_limits(xlims_epoch_loss[0], xlims_epoch_loss[1],
                                   data_epoch_min, data_epoch_max, clip=(1.0, None), pad=0.00)
    if el_lo is not None or el_hi is not None:
        ax_epoch_loss.set_xlim(el_lo, el_hi)

    ea_lo, ea_hi = _resolve_limits(xlims_epoch_acc[0], xlims_epoch_acc[1],
                                   data_epoch_min, data_epoch_max, clip=(1.0, None), pad=0.00)
    if ea_lo is not None or ea_hi is not None:
        ax_epoch_acc.set_xlim(ea_lo, ea_hi)

    tl_lo, tl_hi = _resolve_limits(xlims_time_loss[0], xlims_time_loss[1],
                                   data_time_min, data_time_max, clip=(0.0, None), pad=0.00)
    if tl_lo is not None or tl_hi is not None:
        ax_time_loss.set_xlim(tl_lo, tl_hi)

    ta_lo, ta_hi = _resolve_limits(xlims_time_acc[0], xlims_time_acc[1],
                                   data_time_min, data_time_max, clip=(0.0, None), pad=0.00)
    if ta_lo is not None or ta_hi is not None:
        ax_time_acc.set_xlim(ta_lo, ta_hi)

    # 9. Adjust layout and save the figure
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.93])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"📈 Plot saved → {os.path.abspath(out_path)}")

# ----------------------------- CLI -------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot results saved by the Muon comparison scripts "
            "(CIFAR-10/CifarNet, CIFAR-100/ResNet-18, "
            "Tiny-ImageNet-200/WideResNet-28-10, or FineWeb/NanoGPT)."
        )
    )
    parser.add_argument("--results", type=str, required=True,
                        help="Path to results.json produced by the training script")
    parser.add_argument("--outfile", type=str, default=None,
                        help="Output pdf path. Default: derived from results_dir + dataset/model if available.")

    # Y-axis zoom options
    parser.add_argument("--ylims-loss", type=str, default="none",
                        help="Loss y-lims '<low>,<high>' (e.g., '0.9,1.2' or 'min,1.0').")
    parser.add_argument("--ylims-acc", type=str, default="none",
                        help="Accuracy y-lims '<low>,<high>'. Accepts fractions (0.8,1.0) or percents (80,95).")

    # X-axis zoom options (independent per panel)
    parser.add_argument("--xlims-epoch-loss", type=str, default="none",
                        help="Train-loss epoch x-lims '<low>,<high>' (e.g., '12,max').")
    parser.add_argument("--xlims-time-loss", type=str, default="none",
                        help="Train-loss time x-lims '<low>,<high>' (e.g., '20,max' or '20s,max').")
    parser.add_argument("--xlims-epoch-acc", type=str, default="none",
                        help="Test/Val-acc epoch x-lims '<low>,<high>' (optional).")
    parser.add_argument("--xlims-time-acc", type=str, default="none",
                        help="Test/Val-acc time x-lims '<low>,<high>' (optional).")

    # Fonts (optional)
    parser.add_argument("--axis-label-size", type=int, default=22)
    parser.add_argument("--legend-size", type=int, default=24)

    args = parser.parse_args()

    # Load results
    with open(args.results, "r") as f:
        results = json.load(f)

    # Try to load metadata (for dataset/model info)
    meta = {}
    meta_path = os.path.join(os.path.dirname(os.path.abspath(args.results)), "metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    dataset = meta.get("dataset")
    model = meta.get("model")

    if dataset and model:
        suptitle = f"{dataset} / {model}"
    elif dataset:
        suptitle = dataset
    elif model:
        suptitle = model
    else:
        suptitle = ""

    # Mapping to nicer labels
    name_mapping = {
        "Muon (steps=0)": "Muon(q=0)",
        "Muon (steps=1)": "Muon(q=1)",
        "Muon (steps=2)": "Muon(q=2)",
        "Muon (steps=3)": "Muon(q=3)",
        "Muon with SVD": "Muon(SVD)",
        "SGD with Momentum": "SGD-M",
        "Muon-NS Poly (fit 3.4445,-4.7750,2.0315)": "Ad-hoc Poly (degree 2)",
        "Muon-NS Poly (canonical 1.875,-1.25,0.375)": "Canonical Poly (degree 2)",
    }

    renamed_results = {
        name_mapping.get(key, key): value
        for key, value in results.items()
    }

    # Drop unwanted methods
    drop_keys = {"Muon(q=0)"}
    filtered_results = {
        k: v for k, v in renamed_results.items()
        if k not in drop_keys
    }
    
    # Decide outfile
    out_path = args.outfile
    if out_path is None:
        out_dir = os.path.dirname(os.path.abspath(args.results))

        def _slug(s: str | None) -> str | None:
            if not s:
                return None
            # simple slug: spaces, slashes, parens and colons → underscores
            bad_chars = [" ", "/", "(", ")", ":"]
            for ch in bad_chars:
                s = s.replace(ch, "_")
            return s

        core = "muon_steps_vs_svd_vs_sgdm_large"
        ds_slug = _slug(dataset)
        model_slug = _slug(model)
        if ds_slug:
            core += f"_{ds_slug}"
        if model_slug:
            core += f"_{model_slug}"
        out_path = os.path.join(out_dir, f"{core}.pdf")

    # Parse y-limit strings
    loss_lo_tok, loss_hi_tok = _parse_lims(args.ylims_loss)
    acc_lo_tok,  acc_hi_tok  = _parse_lims(args.ylims_acc)

    # Treat numbers > 1 for accuracy as percents
    def _percentify(tok):
        if isinstance(tok, (int, float)) and tok > 1.0:
            return float(tok) / 100.0
        return tok
    acc_lo_tok = _percentify(acc_lo_tok)
    acc_hi_tok = _percentify(acc_hi_tok)

    # Parse x-limit strings
    el_lo, el_hi = _parse_lims(args.xlims_epoch_loss)
    tl_lo, tl_hi = _parse_lims(args.xlims_time_loss)
    ea_lo, ea_hi = _parse_lims(args.xlims_epoch_acc)
    ta_lo, ta_hi = _parse_lims(args.xlims_time_acc)

    plot_comparison(
        filtered_results,
        out_path,
        suptitle=suptitle,
        axis_label_size=args.axis_label_size,
        legend_size=args.legend_size,
        ylims_loss=(loss_lo_tok, loss_hi_tok),
        ylims_acc=(acc_lo_tok,  acc_hi_tok),
        xlims_epoch_loss=(el_lo, el_hi),
        xlims_time_loss=(tl_lo, tl_hi),
        xlims_epoch_acc=(ea_lo, ea_hi),
        xlims_time_acc=(ta_lo, ta_hi),
    )

if __name__ == "__main__":
    main()
