import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

def load_data(csvfile):
    df = pd.read_csv(csvfile)
    return df

def parse_fileinfo(filename):
    # Example filename: hns_4x4x1_cage15_main_Acsc_spcomm.out
    m = re.match(r"hns_(\d+)x(\d+)x(\d+)_(\w+)_main", filename)
    if not m:
        return None, None
    gridrows, gridcols, gridz, matrixname = m.groups()
    return int(gridrows), int(gridcols), int(gridz), matrixname

def make_heatmap(df, metric, matrix, gridrows, gridcols, dataset, outfile):
    """
    df: filtered dataframe for given dataset (matrix + gridsize)
    metric: 'comp_rate' or 'comp_time'
    matrix: 'A' or 'B'
    """
    dsub = df[(df["metric"] == metric) & (df["matrix"] == matrix)]
    if dsub.empty:
        return

    # For comp_rate: only round 0
    if metric == "comp_rate":
        dsub = dsub[dsub["round"] == 0].copy()
        dsub["ratio"] = dsub["endnnz"] / dsub["startnnz"]
        dsub["size"] = (dsub["endnnz"] * 4) / (1024 * 1024) # *4 is for bytes, / for MB

    # Build square matrices
    valmat = np.full((gridrows, gridcols), np.nan)
    bytemat = np.full((gridrows, gridcols), np.nan)

    for _, row in dsub.iterrows():
        i = int(row["process"] // gridcols)
        j = int(row["target"] % gridcols)
        valmat[i, j] = row["ratio"]
        bytemat[i, j] = row["size"]

    # Labels: combine ratio + size
    labels = np.empty_like(valmat, dtype=object)
    for i in range(gridrows):
        for j in range(gridcols):
            if not np.isnan(valmat[i, j]):
                labels[i, j] = f"{valmat[i,j]:.2f}\n{int(bytemat[i,j])} MB"
            else:
                labels[i, j] = ""

    # Plot heatmap with only `ratio` driving colors
    plt.figure(figsize=(8, 6))
    mylabel = "%s in [0.0-1.1] + comp_size in MB (on both lower is better)" % metric if metric == "comp_rate" else metric
    sns.heatmap(valmat, annot=labels, fmt="", cmap="viridis",
                cbar_kws={'label': mylabel}, linewidths=0.5, linecolor="gray")

    plt.title(f"{dataset} {gridrows}x{gridcols} - {metric} (Matrix {matrix})")
    plt.xlabel("Target process")
    plt.ylabel("Source process")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


def main(csvfile, outdir="compression"):
    df = load_data(csvfile)
    Path(outdir).mkdir(exist_ok=True)

    for fname in df["file"].unique():
        gridrows, gridcols, gridz, dataset = parse_fileinfo(fname)
        if not gridrows or not gridcols or not dataset:
            continue
        dfile = df[df["file"] == fname]

        # for metric in ["comp_rate", "comp_time"]:
        for metric in ["comp_rate"]:
            for matrix in ["A", "B"]:
                outname = f"{dataset}_{gridrows}x{gridcols}x{gridz}_{metric}_M{matrix}.png"
                make_heatmap(dfile, metric, matrix, gridrows, gridcols, dataset, Path(outdir)/outname)

if __name__ == "__main__":
    main("new_compression.csv")

