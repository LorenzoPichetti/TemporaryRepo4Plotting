import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np
import re

def extract_rank(fname):
    """
    Extract the process number from filenames like:
    "cage15_process0_operandA_4x4x1_targets"
    """
    # Match 'process' followed by one or more digits
    m = re.search(r'process(\d+)', fname)
    return int(m.group(1)) if m else -1  # fallback if not found

def merge_plots(dataset, grid, outdir="overall"):
    """
    Merge:
      - Top row: matrix_internode plot (left) + heatmaps A/B (right)
      - Bottom: rank_intranode plots laid out in two grids:
        first all operand A plots, then operand B plots
    """
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    # Input folders
    heatdir = Path("compression")
    rankdir = Path("rank_internode")
    matdir  = Path("matrix_internode")

    # Collect images
    heatA_file     = heatdir / f"{dataset}_{grid}_MA.png"
    heatB_file     = heatdir / f"{dataset}_{grid}_MB.png"
    internode_file = matdir  / f"{dataset}_{grid}_barplot.png"

    # Rank plots for A and B
    rankA_files = sorted(rankdir.glob(f"{dataset}_process*_operandA_{grid}_targets.png"))
    rankB_files = sorted(rankdir.glob(f"{dataset}_process*_operandB_{grid}_targets.png"))
    rankA_files = sorted(rankA_files, key=lambda p: extract_rank(str(p)))
    rankB_files = sorted(rankB_files, key=lambda p: extract_rank(str(p)))

    # Load as PIL images
    def load_img(path):
        if path.exists():
            return Image.open(path)
        return None

    internode_img = load_img(internode_file)
    heatA_img     = load_img(heatA_file)
    heatB_img     = load_img(heatB_file)
    rankA_imgs    = [load_img(f) for f in rankA_files if load_img(f) is not None]
    rankB_imgs    = [load_img(f) for f in rankB_files if load_img(f) is not None]

    # Safety check
    if internode_img is None or heatA_img is None or heatB_img is None:
        print(f"Missing images for {dataset} {grid}, skipping [{internode_img}, {heatA_img}, {heatB_img}].")
        return

    # --- Build figure ---
    fig = plt.figure(figsize=(50, 50), constrained_layout=True)

    # Use subfigures: top for internode+heatmaps, bottom for ranks
    subfigs = fig.subfigures(2, 1, height_ratios=[1, 2])

    # ------------------- TOP -------------------
    top = subfigs[0].subplots(1, 2)

    # Internode (left)
    ax1 = top[0]
    ax1.imshow(internode_img)
    ax1.axis("off")
    ax1.set_title("Internode")

    # Heatmaps (right, A and B side by side)
    ax2 = top[1]
    combined_width = heatA_img.width + heatB_img.width
    combined_height = max(heatA_img.height, heatB_img.height)
    combined = Image.new("RGB", (combined_width, combined_height), (255, 255, 255))
    combined.paste(heatA_img, (0, 0))
    combined.paste(heatB_img, (heatA_img.width, 0))
    ax2.imshow(combined)
    ax2.axis("off")
    ax2.set_title("Heatmaps (A / B)")

    # ------------------- BOTTOM -------------------
    if rankA_imgs or rankB_imgs:
        try:
            gridrows, gridcols = map(int, grid.split("x")[:2])
        except Exception:
            gridrows, gridcols = 1, max(len(rankA_imgs), len(rankB_imgs))  # fallback

        nprocs = max(len(rankA_imgs), len(rankB_imgs))
        nrows_per_operand = int(np.ceil(nprocs / gridcols))
        total_rows = 2 * nrows_per_operand  # A block + B block

        axs = subfigs[1].subplots(total_rows, gridcols, squeeze=False)

        # Fill block A
        for i, img in enumerate(rankA_imgs):
            r, c = divmod(i, gridcols)
            axs[r, c].imshow(img)
            axs[r, c].axis("off")
            axs[r, c].set_title(f"Rank {i} (A)")
        for j in range(len(rankA_imgs), nrows_per_operand * gridcols):
            r, c = divmod(j, gridcols)
            axs[r, c].axis("off")

        # Fill block B (shifted down by nrows_per_operand)
        for i, img in enumerate(rankB_imgs):
            r, c = divmod(i, gridcols)
            r += nrows_per_operand
            axs[r, c].imshow(img)
            axs[r, c].axis("off")
            axs[r, c].set_title(f"Rank {i} (B)")
        for j in range(len(rankB_imgs), nrows_per_operand * gridcols):
            r, c = divmod(j, gridcols)
            r += nrows_per_operand
            axs[r, c].axis("off")

    else:
        print(f"No rank_intranode plots for {dataset} {grid}.")

    # ------------------- SAVE -------------------
    fig.suptitle(f"Summary plots for {dataset} {grid}", fontsize=20)
    outfile = outdir / f"{dataset}_{grid}_summary.png"
    plt.savefig(outfile, bbox_inches="tight", dpi=100)
    plt.close()
    print(f"Saved {outfile}")


def main():
    datasets = [("nlpkkt160", "4x4x1"),
                ("cage15", "4x4x1"),
                ("HV15R", "4x4x1"),
                ("dielFilterV3real", "4x4x1"),
                ("ldoor", "4x4x1")]
    for dataset, grid in datasets:
        merge_plots(dataset, grid)


if __name__ == "__main__":
    main()
