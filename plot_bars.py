import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path

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

import matplotlib.patches as mpatches

def make_barplot(df, grid, matrix, outdir, compute):
    # ----- Aggregation phase -----
    # Step 1: average over rounds
    df_roundavg = df.groupby(
        ["process", "target", "operand", "config"], as_index=False
    )[["compression", "communication"]].mean()

    # Step 2: sum over targets (keeping process/operand/config)
    df_sum = df_roundavg.groupby(
        ["process", "operand", "config"], as_index=False
    )[["compression", "communication"]].sum()
    # -----------------------------

    # Get order of processes and configs
    proc_order = sorted(df_sum["process"].unique(), key=int)
    config_order = ["--impl get", "--impl main", "--impl main --Acsc", "--impl main --Acsc --spcomm"]

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


def make_target_plots(df, grid, matrix, outdir, compute):
    """
    For each process rank:
      - x-axis = combined operand+target (e.g., A0, A1, ..., B0, B1, ...)
      - for each tick, multiple bars = different configs
      - bars stacked into communication + compression
    """

    # Average over rounds
    dfavg = df.groupby(["process", "target", "operand", "config"], as_index=False)[["compression", "communication"]].mean()

    proc_order = sorted(dfavg["process"].unique(), key=int)
    target_order = sorted(dfavg["target"].unique(), key=int)
    config_order = ["--impl get", "--impl main", "--impl main --Acsc", "--impl main --Acsc --spcomm"]
    cmap = plt.get_cmap("tab10")

    for proc in proc_order:
        dproc = dfavg[dfavg["process"] == proc]

        # Build combined xtick labels
        xticks = []
        for op in ["A", "B"]:
            for t in target_order:
                xticks.append(f"{op}{t}")

        x = np.arange(len(xticks))  # tick positions
        width = 0.15  # width per config bar

        plt.figure(figsize=(14, 6))

        for i, cfg in enumerate(config_order):
            dcfg = dproc[dproc["config"] == cfg]
            y_comm, y_comp = [], []

            for op in ["A", "B"]:
                for t in target_order:
                    row = dcfg[(dcfg["operand"] == op) & (dcfg["target"] == t)]
                    if not row.empty:
                        y_comm.append(row["communication"].values[0])
                        y_comp.append(row["compression"].values[0])
                    else:
                        y_comm.append(0)
                        y_comp.append(0)

            xpos = x + i * width - (len(config_order)-1) * width / 2
            color = cmap(i)

            # Communication
            plt.bar(xpos, y_comm, width, color=color, edgecolor="black", label=cfg if i == 0 else None)
            # Compression stacked
            plt.bar(xpos, y_comp, width, bottom=y_comm, color=color, hatch="///", edgecolor="black")

        plt.xticks(x, xticks, rotation=45)
        plt.xlabel("Operand + Target process")
        plt.ylabel("Average time")
        plt.title(f"Process {proc} - Internode comm breakdown ({grid}) - {compute}")
        plt.tight_layout()

        # Legends
        config_patches = [mpatches.Patch(color=cmap(i), label=cfg) for i, cfg in enumerate(config_order)]
        metric_patches = [
            mpatches.Patch(facecolor="white", edgecolor="black", label="Communication"),
            mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="Compression")
        ]

        leg1 = plt.legend(handles=config_patches, title="Configuration", loc="upper left", bbox_to_anchor=(1.02, 1))
        plt.gca().add_artist(leg1)
        plt.legend(handles=metric_patches, title="Metric", loc="lower left", bbox_to_anchor=(1.02, 0))

        outfile = Path(outdir) / f"{matrix}_process{proc}_{grid}_targets.png"
        plt.savefig(outfile, bbox_inches="tight")
        plt.close()



def main(csvfile, outdir="internode"):
    Path(outdir).mkdir(exist_ok=True)

    df = load_data(csvfile)
    df[["grid", "matrix", "config", "compute"]] = df["file"].apply(
        lambda f: pd.Series(parse_params(f))
    )

    # Loop over dataset = {matrix, grid}
    for (matrix, grid, compute), dsub in df.groupby(["matrix", "grid", "compute"]):
        make_barplot(dsub, grid, matrix, "matrix_" + outdir, compute)
        make_target_plots(dsub, grid, matrix, "rank_" + outdir, compute)

if __name__ == "__main__":
    main("internode.csv")

