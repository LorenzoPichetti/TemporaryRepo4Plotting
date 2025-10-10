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

            # Internode communication
            m = re.match(r'<\[p (\d+), t (\d+), m ([AB])\]>\[internode_comm\(comp\+comm\+size\)\] ([0-9.]+) ms, ([0-9.]+) ms, (\d+) B, (\d+) B', line)
            if m:
                rank, target, mat, comptime, commtime, origsize, compsize = m.groups()
                results.append({
                    "file": filepath.name,
                    "round": run_i,
                    "process": int(rank),
                    "target": int(target),
                    "metric": "internode_time",
                    "operand": mat,
                    "compression": float(comptime),
                    "communication": float(commtime),
                    "orig_size": int(origsize),
                    "compressed_size": int(compsize)
                })

    return results

def main(input_dir: str, output_csv: str):
    input_path = Path(input_dir)
    all_results = []

    for file in input_path.glob("*.out"):
        print("processing file ", file)
        all_results.extend(parse_file(file))

    keys = ["file", "round", "process", "target", "metric", "operand", "compression", "communication", "orig_size", "compressed_size"]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"Saved internode results to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python parse_internode.py <input_dir> <output_csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

