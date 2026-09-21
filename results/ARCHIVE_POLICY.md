# results/ — Paper Artifact Archive (append-only)

**Policy: NOTHING in this tree is ever deleted.** Every CSV, plot, JSON, and
trace from every run — including pilots, nulls, and debug runs — is kept and
organized per milestone. If a rerun supersedes an older file, the old one is
archived under `_archive/` with a timestamp, never removed.

The paper (§64, §86, §87) draws every figure/table from this tree, so each
milestone keeps `csv/`, `plots/`, and `json/` outputs together with the exact
notebook/script version that produced them (recorded in `RUN_MANIFEST.md`).

## Layout

```
results/
├── m0_exactness/            exact decomposition (exactness_report.csv)
├── m1_oracle_sparsity/      oracle sweep (oracle_ffn.csv, summary tables, plots)
├── m2_task_specialization/  contiguous-block specialization (json, heatmap png)
├── m3_neuron_clustering/    clustering + specialization re-test (all arms)
├── m3b_granularity_option2/ finer-k + token-space arms (option 2 test)
├── csv/  plots/  traces/    legacy working dirs (kept; mirrored into m*/ folders)
├── logs/                    per-run provenance JSONL (§65 reproducibility)
└── _archive/                superseded outputs, timestamped, never deleted
```

## Run manifest

Every execution appends one line to `results/RUN_MANIFEST.md`: date, milestone,
what was run, key numbers, output paths, and the notebook/script used. The
paper's reproduction guide is generated from this manifest plus the archived
files.
