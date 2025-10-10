import re
import csv
import sys
from pathlib import Path

def sanitize(line: str) -> str:
    line = re.sub(r'^\x1b\[[0-9;]*m', '', line)
    line = re.sub(r'\x1b\[0m$', '', line)
    return line.strip()

def parse_file(filepath: Path):
    results = []
    run_i = -1

    with open(filepath, "r") as f:
        for raw in f:
            line = sanitize(raw)

            if line.startswith("STARTING spgemm round"):
                run_i += 1

            # Compression rate
            m = re.match(r'<\[p (\d+), t (\d+), m ([AB])\]>\[internode_comm\(comp\+comm\+size\)\] [0-9.]+ ms, [0-9.]+ ms, ([0-9]+) B, ([0-9]+) B', line)
            if m:
                rank, target, mat, startnnz, endnnz = m.groups()
                results.append({
                    "file": filepath.name,
                    "round": run_i,
                    "process": int(rank),
                    "target": int(target),
                    "metric": "comp_rate",
                    "matrix": mat,
                    "startnnz": int(startnnz),
                    "endnnz": int(endnnz)
                })

            # Compression time
            m = re.match(r'<\[p (\d+), t (\d+), m ([AB])\]>\[compression_time\] ([0-9.]+) ms', line)
            if m:
                rank, target, mat, ctime = m.groups()
                results.append({
                    "file": filepath.name,
                    "round": run_i,
                    "process": int(rank),
                    "target": int(target),
                    "metric": "comp_time",
                    "matrix": mat,
                    "time": float(ctime)
                })

    return results

def main(input_dir: str, output_csv: str):
    input_path = Path(input_dir)
    all_results = []

    for file in input_path.glob("*.out"):
        if "spcomm" not in file.name:
            continue
        all_results.extend(parse_file(file))

    keys = ["file", "round", "process", "target", "metric", "matrix", "startnnz", "endnnz", "time"]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"Saved compression results to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python parse_compression.py <input_dir> <output_csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

