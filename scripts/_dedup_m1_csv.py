"""One-time dedup repair for the 8B M1 CSV.

The old resume-scan keyed on (layer, CATEGORY) instead of (layer, PROMPT) and
never matched, so every restart re-measured ALL units, appending duplicate
sweeps. Each sweep writes retain_fraction in strictly descending order
(1.0, 0.7, ..., 0.05), so within a key any increase marks a new sweep.
Keeps only the FIRST sweep per (layer, prompt) key. Backs up the original
into results/_archive/ per the archive policy.
"""
import csv
import shutil
from collections import Counter, defaultdict
from pathlib import Path

csv_path = Path(r"e:\Project\SliceWeave\Qwen3-8B_Results\results\m1_oracle_sparsity\csv\oracle_ffn.csv")
archive = csv_path.parents[2] / "_archive"
archive.mkdir(parents=True, exist_ok=True)

with open(csv_path, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    header, rows = reader.fieldnames, list(reader)
print("rows before:", len(rows))

key = lambda r: (r["layer"], r["prompt"])
seqs = defaultdict(list)                       # key -> [row, ...] (current sweep)
dup_keys, dup_rows = set(), 0
for r in rows:
    seq = seqs[key(r)]
    if seq and float(r["retain_fraction"]) > float(seq[-1]["retain_fraction"]):
        dup_keys.add(key(r))                   # a new sweep begins
        dup_rows += len(seq)
    seq.append(r)

print("keys with duplicate sweeps:", len(dup_keys))
if dup_keys:
    backup = archive / f"oracle_ffn_prededup_{csv_path.stat().st_mtime_ns}.csv"
    shutil.copy2(csv_path, backup)
    print("backup:", backup)

    keep, seen_key = [], set()
    for r in rows:
        k = key(r)
        seq = seqs[k]
        is_new_sweep_start = seq and seq[0] is r
        if k in dup_keys and k in seen_key:
            continue                            # drop everything after first sweep
        keep.append(r)
        if is_new_sweep_start or len(seqs[k]) == 1:
            pass
        # mark key seen once its first sweep ends (next row of same key that
        # starts a new sweep is skipped by the check above)
        nxt = None
        seen_key.add(k) if False else None
    # simpler: rebuild per key
    keep = []
    for k, seq in seqs.items():
        # split seq into sweeps at increases; keep first sweep
        first = [seq[0]]
        for prev, cur in zip(seq, seq[1:]):
            if float(cur["retain_fraction"]) > float(prev["retain_fraction"]):
                break
            first.append(cur)
        keep.extend(first)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([[r[c] for c in header] for r in keep])
    print(f"rows after: {len(keep)}  (removed {len(rows) - len(keep)})")

# verification
with open(csv_path, encoding="utf-8") as f:
    rows2 = list(csv.DictReader(f))
per_key = Counter(key(r) for r in rows2)
frac_sets = defaultdict(set)
for r in rows2:
    frac_sets[key(r)].add(round(float(r["retain_fraction"]), 2))
WANT = {1.0, 0.7, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05}
print("distinct keys:", len(per_key))
print("rows-per-key histogram:", dict(sorted(Counter(per_key.values()).items())))
incomplete = [k for k, fs in frac_sets.items() if fs != WANT]
print("keys missing any fraction:", len(incomplete))
complete = sum(1 for fs in frac_sets.values() if fs == WANT)
print("complete units (all 9 fractions):", complete)
