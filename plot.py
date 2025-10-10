import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path

import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

config_order = ["--impl get", "--impl main", "--impl main --Acsc", "--impl main --Acsc --spcomm"]

def load_data(csvfile):
    return pd.read_csv(csvfile)

def parse_params(filename):
    """
    Parse filename of format:
      hns_{grid}_{matrix}_{impl}_{Acsc}_{spcomm}.out
    Return (grid, matrix, config_string).
    """
    m = re.match(r"hns_([^_]+)_(\w+)_(\w+)_(\w+)_(\w+)_(\w+)\.out", filename)
    if not m:
        return None, None, "unknown"
    grid, matrix, impl, acsc, spcomm, skip = m.groups()

    parts = [f"--impl {impl}"]
    if acsc != "none":
        parts.append("--Acsc")
    if spcomm != "none":
        parts.append("--spcomm")

    if skip != "none":
        compute_type = "skip computation"
    else:
        compute_type = "kokkos computation"

    return grid, matrix, " ".join(parts), compute_type

def preprocess_dataframe(df):
    # ----- Aggregation phase -----
    # Step 0: drop all rows where round == 0 (warm-up)
    df_filtered = df[df["round"] != 0]

    # Step 1: consistency check
    check = df_filtered.groupby(
        ["process", "target", "operand", "config"]
    )[["orig_size", "compressed_size"]].nunique()

    bad = check[(check["orig_size"] > 1) | (check["compressed_size"] > 1)]
    if not bad.empty:
        raise ValueError(f"Inconsistent orig/compressed sizes detected:\n{bad}")

    # Step 2: average over rounds
    df_roundavg = df_filtered.groupby(
        ["process", "target", "operand", "config"], as_index=False
    )[["compression", "communication", "orig_size", "compressed_size"]].mean()
    # Due to the previous test, we know that the mean of "orig_size" and "compressed_size"
    # is exactly the unique value in these fields (i.e. variance 0).

    # Step 3: sum over targets (keeping process/operand/config)
    df_sum = df_roundavg.groupby(
        ["process", "operand", "config"], as_index=False
    )[["compression", "communication"]].sum()
    # -----------------------------

    return df_roundavg, df_sum

def make_barplot(df_sum, grid, matrix, outdir, compute):

    # Get order of processes and configs
    proc_order = sorted(df_sum["process"].unique(), key=int)

    # Prepare bar positions
    import numpy as np
    x = np.arange(len(proc_order))  # one tick per process
    width = 0.2  # width of each config’s bar

    plt.figure(figsize=(14, 6))

    # Pick a colormap for configs
    cmap = plt.get_cmap("tab10")  # categorical palette

    for i, cfg in enumerate(config_order):
        dcfg = df_sum[df_sum["config"] == cfg]
        # align with proc_order
        y_comp = [dcfg.loc[dcfg["process"] == p, "compression"].values[0] if p in dcfg["process"].values else 0 for p in proc_order]
        y_comm = [dcfg.loc[dcfg["process"] == p, "communication"].values[0] if p in dcfg["process"].values else 0 for p in proc_order]

        xpos = x + i * width - (len(config_order)-1) * width / 2
        color = cmap(i)

        # Communication = base (plain fill)
        plt.bar(xpos, y_comm, width, label=f"{cfg}", color=color, edgecolor="black")

        # Compression stacked on top, same color but hatched
        plt.bar(xpos, y_comp, width, bottom=y_comm, color=color, hatch="///", edgecolor="black")

    # Axis formatting
    plt.xticks(x, proc_order)
    plt.xlabel("MPI process rank")
    plt.ylabel("Aggregated time (ms)")
    plt.title(f"Internode communication - {matrix} ({grid}) - {compute}")
    plt.tight_layout()

    # Build legends
    # 1) Legend for configs (colors)
    config_patches = [
        mpatches.Patch(color=cmap(i), label=cfg)
        for i, cfg in enumerate(config_order)
    ]
    # 2) Legend for metrics (textures)
    metric_patches = [
        mpatches.Patch(facecolor="white", edgecolor="black", label="Communication"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="Compression")
    ]

    leg1 = plt.legend(handles=config_patches, title="Configuration", loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.gca().add_artist(leg1)  # keep first legend
    plt.legend(handles=metric_patches, title="Metric", loc="lower left", bbox_to_anchor=(1.02, 0))

    # Save
    outfile = Path(outdir) / f"{matrix}_{grid}_barplot.png"
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()


def make_target_plots(dfavg, grid, matrix, outdir, compute):
    """
    For each process rank and operand (A or B):
      - x-axis = target process
      - grouped bars for different configs
      - bars stacked into communication + compression
      - secondary y-axis = orig_size and compressed_size line plots
    """

    proc_order = sorted(dfavg["process"].unique(), key=int)
    target_order = sorted(dfavg["target"].unique(), key=int)
    cmap = plt.get_cmap("tab10")

    for proc in proc_order:
        for op in ["A", "B"]:
            dsel = dfavg[(dfavg["process"] == proc) & (dfavg["operand"] == op)]
            if dsel.empty:
                continue

            x = np.arange(len(target_order))  # tick positions = targets only
            width = 0.15

            fig, ax1 = plt.subplots(figsize=(14, 6))

            for i, cfg in enumerate(config_order):
                dcfg = dsel[dsel["config"] == cfg]
                y_comm, y_comp = [], []

                for t in target_order:
                    row = dcfg[dcfg["target"] == t]
                    if not row.empty:
                        y_comm.append(row["communication"].values[0])
                        y_comp.append(row["compression"].values[0])
                    else:
                        y_comm.append(0)
                        y_comp.append(0)

                xpos = x + i * width - (len(config_order)-1) * width / 2
                color = cmap(i)

                ax1.bar(xpos, y_comm, width, color=color, edgecolor="black", label=cfg if i == 0 else None)
                ax1.bar(xpos, y_comp, width, bottom=y_comm, color=color, hatch="///", edgecolor="black")

            # Secondary y-axis for sizes
            ax2 = ax1.twinx()
            # Average over configs (sizes should be consistent anyway)
            size_avg = dsel.groupby("target")[["orig_size", "compressed_size"]].mean().reset_index()

            ax2.plot(x, size_avg["orig_size"], marker="o", color="red", label="orig_size")
            ax2.plot(x, size_avg["compressed_size"], marker="s", color="blue", label="compressed_size")

            ax1.set_xticks(x)
            ax1.set_xticklabels([f"{op}{t}" for t in target_order], rotation=45)

            ax1.set_xlabel("Target process")
            ax1.set_ylabel("Average time")
            ax2.set_ylabel("Message size (bytes)")

            ax1.set_title(f"Process {proc}, Operand {op} - Internode comm breakdown ({grid}) - {compute}")
            fig.tight_layout()

            # Legends
            config_patches = [mpatches.Patch(color=cmap(i), label=cfg) for i, cfg in enumerate(config_order)]
            metric_patches = [
                mpatches.Patch(facecolor="white", edgecolor="black", label="Communication"),
                mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="Compression")
            ]
            size_lines = [
                Line2D([0], [0], color="red", marker="o", label="orig_size"),
                Line2D([0], [0], color="blue", marker="s", label="compressed_size"),
            ]

            leg1 = ax1.legend(handles=config_patches, title="Configuration", loc="upper left", bbox_to_anchor=(1.02, 1))
            ax1.add_artist(leg1)
            leg2 = ax1.legend(handles=metric_patches + size_lines, title="Metric", loc="lower left", bbox_to_anchor=(1.02, 0))

            outfile = Path(outdir) / f"{matrix}_process{proc}_operand{op}_{grid}_targets.png"
            plt.savefig(outfile, bbox_inches="tight")
            plt.close()

def human_readable_size(num_bytes, suffix="B"):
    """
    Convert a size in bytes to a human-readable string
    using binary units (KiB, MiB, GiB, ...).
    """
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f} {unit}{suffix}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} Y{suffix}"  # for very large sizes


def make_heatmap(df, operand, gridrows, gridcols, dataset, outfile):
    """
    df: filtered dataframe for given dataset (matrix + gridsize)
    metric: 'comp_rate' or 'comp_time'
    matrix: 'A' or 'B'
    """

    # Build square matrices
    valmat   = np.full((gridrows, gridcols), np.nan)
    byteorig = np.full((gridrows, gridcols), "", dtype=object)
    bytecomp = np.full((gridrows, gridcols), "", dtype=object)

    for _, row in df.iterrows():
        i = int(row["process"] // gridcols)
        j = int(row["process"] %  gridcols)
        byteorig[i, j] = human_readable_size(int(row["orig_size"]))
        bytecomp[i, j] = human_readable_size(int(row["compressed_size"]))
        if (row["orig_size"] != 0):
            valmat[i, j] = row["compressed_size"] / row["orig_size"]
        else:
            valmat[i, j] = 1.0

    # Labels: combine ratio + size
    labels = np.empty_like(valmat, dtype=object)
    for i in range(gridrows):
        for j in range(gridcols):
            if not np.isnan(valmat[i, j]):
                labels[i, j] = f"{valmat[i,j]:.2f}\n{byteorig[i,j]}-{bytecomp[i,j]}"
            else:
                labels[i, j] = ""

    # Plot heatmap with only `ratio` driving colors
    plt.figure(figsize=(8, 6))
    mylabel = "rate in [0.0-1.1] + orig_size-comp_size in (on both lower is better)"
    sns.heatmap(valmat, annot=labels, fmt="", cmap="viridis",
                cbar_kws={'label': mylabel}, linewidths=0.5, linecolor="gray")

    plt.title(f"{dataset} {gridrows}x{gridcols} - average compression (Operand {operand})")
    plt.xlabel("Target process")
    plt.ylabel("Source process")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


def main(csvfile, outdir="internode"):
    Path(outdir).mkdir(exist_ok=True)

    compressionoutdir="compression"
    Path(compressionoutdir).mkdir(exist_ok=True)

    df = load_data(csvfile)
    df[["grid", "matrix", "config", "compute"]] = df["file"].apply(
        lambda f: pd.Series(parse_params(f))
    )

    # Loop over dataset = {matrix, grid}
    for (matrix, grid, compute), dsub in df.groupby(["matrix", "grid", "compute"]):
        dfavg, df_sum = preprocess_dataframe(dsub)
        make_barplot(df_sum, grid, matrix, "matrix_" + outdir, compute)
        make_target_plots(dfavg, grid, matrix, "rank_" + outdir, compute)


        m = re.match(r"(\d+)x(\d+)x\d+", grid)
        if not m:
            return None, None
        gridrows, gridcols = m.groups()

        for operand in ["A", "B"]:
            dsub = dfavg[dfavg["operand"] == operand]
            outname = f"{matrix}_{grid}_M{operand}.png"
            make_heatmap(dsub, operand, int(gridrows), int(gridcols), matrix, Path(compressionoutdir)/outname)

if __name__ == "__main__":
    main("internode.csv")

