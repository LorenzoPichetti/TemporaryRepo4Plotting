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
    m = re.match(r"hns_(\d+)x(\d+)x\d+_(\w+)_main", filename)
    if not m:
        return None, None
    gridrows, gridcols, matrixname = m.groups()
    return int(gridrows), int(gridcols), matrixname

def make_heatmap(df, metric, matrix, gridrows, gridcols, dataset, outfile):
    """
    df: filtered dataframe for given dataset (matrix + gridsize)
    metric: 'comp_rate' or 'comp_time'
    matrix: 'A' or 'B'
    """
    # Filter by matrix + metric
    dsub = df[(df["metric"] == metric) & (df["matrix"] == matrix)]
    if dsub.empty:
        return

    # For comp_rate: only round 0
    if metric == "comp_rate":
        dsub = dsub[dsub["round"] == 0]
        dsub["value"] = dsub["endnnz"] / dsub["startnnz"]
    else:  # comp_time
        # average over rounds
        dsub = dsub.groupby(["process", "target"], as_index=False)["time"].mean()
        dsub["value"] = dsub["time"]

    # Build square matrix
    nproc = max(gridrows, gridcols)
    mat = np.full((gridrows, gridcols), 0.0)
    for _, row in dsub.iterrows():
        mat[int(row["process"]//gridcols), int(row["target"]%gridcols)] += row["value"]

    for i in range(0,gridrows):
        for j in range(0, gridcols):
            mat[i, j] /= gridcols

    plt.figure(figsize=(8, 6))
    if (metric == "comp_rate"):
        mylable = "%s ([0.0-1.1] lower is better)" % metric
    else:
        mylable = "%s ms (overall time)" % metric
    sns.heatmap(mat, annot=True, cmap="viridis", cbar_kws={'label': mylable})
    plt.title(f"{dataset} {gridrows}x{gridcols} - {metric} (Matrix {matrix})")
    plt.xlabel("Target process")
    plt.ylabel("Source process")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

def main(csvfile, outdir="heatmaps"):
    df = load_data(csvfile)
    Path(outdir).mkdir(exist_ok=True)

    for fname in df["file"].unique():
        gridrows, gridcols, dataset = parse_fileinfo(fname)
        if not gridrows or not gridcols or not dataset:
            continue
        dfile = df[df["file"] == fname]

        for metric in ["comp_rate", "comp_time"]:
            for matrix in ["A", "B"]:
                outname = f"{dataset}_{gridrows}x{gridcols}_{metric}_M{matrix}.png"
                make_heatmap(dfile, metric, matrix, gridrows, gridcols, dataset, Path(outdir)/outname)

if __name__ == "__main__":
    main("new_compression.csv")

