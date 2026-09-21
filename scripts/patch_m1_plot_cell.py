"""Replace the M1 plot cell in notebooks/oracle_milestone1.ipynb (id
#VSC-49d9f144 / index 18) with the full three-plot version that mirrors the
4B naming convention. Direct notebook-edit tooling kept dropping or ignoring
the rewrite, so this does a raw JSON patch with verification."""
import json
from pathlib import Path

NB = Path(r"e:\Project\SliceWeave\notebooks\oracle_milestone1.ipynb")

NEW_SOURCE = '''# M1 plots — regenerate ALL THREE from the CSV on disk (4B naming convention).
# Reading the CSV fresh makes this cell deterministic and re-runnable.
import csv as _csv
from collections import defaultdict

by_layer = defaultdict(lambda: defaultdict(list))   # layer -> frac -> [cosine]
cat30 = defaultdict(list)                           # category -> [cosine @30%]
all_by_frac = defaultdict(list)                     # frac -> [cosine]
with open(OUT_CSV, encoding="utf-8") as f:
    for r in _csv.DictReader(f):
        frac = float(r["retain_fraction"]); cos = float(r["cosine"])
        by_layer[r["layer"]].setdefault(frac, []).append(cos)
        all_by_frac[frac].append(cos)
        if round(frac, 2) == 0.30:
            cat30[r["category"]].append(cos)

PLOT_DIR.mkdir(parents=True, exist_ok=True)

# 1) m1_quality_vs_retention.png — per-layer mean cosine vs retention
figA, axA = plt.subplots(figsize=(8, 5))
for layer, d in sorted(by_layer.items()):
    xs = sorted(d)
    axA.plot(xs, [sum(d[x])/len(d[x]) for x in xs], marker="o", ms=4,
             label=f"layer {layer.split('.')[-1]}")
axA.axhline(0.99, ls="--", c="gray", lw=1)
axA.set_xlabel("retention fraction"); axA.set_ylabel("cosine vs dense")
axA.set_title("Qwen3-8B M1: quality vs expert retention (per layer)")
axA.legend(); axA.grid(alpha=0.3)
figA.tight_layout()
pA = PLOT_DIR / "m1_quality_vs_retention.png"
figA.savefig(pA, dpi=150)

# 2) m1_category_robustness_30pct.png — per-category barh @30%
cats = sorted(cat30)
figB, axB = plt.subplots(figsize=(8, 6))
axB.barh(cats, [sum(cat30[c])/len(cat30[c]) for c in cats])
axB.set_xlim(0.95, 1.0); axB.set_title("Qwen3-8B M1: cosine @30% by category")
axB.grid(alpha=0.3, axis="x")
figB.tight_layout()
pB = PLOT_DIR / "m1_category_robustness_30pct.png"
figB.savefig(pB, dpi=150)

# 3) m1_summary_curve.png — overall mean ± std vs retention
xs = sorted(all_by_frac)
means = [sum(all_by_frac[x])/len(all_by_frac[x]) for x in xs]
sds = [ (statistics.stdev(all_by_frac[x]) if len(all_by_frac[x]) > 1 else 0.0) for x in xs ]
figC, axC = plt.subplots(figsize=(8, 5))
axC.plot(xs, means, marker="o", ms=4, label="mean cosine")
axC.fill_between(xs, [m-s for m, s in zip(means, sds)], [m+s for m, s in zip(means, sds)], alpha=0.2, label="±1 std")
axC.axhline(0.99, ls="--", c="gray", lw=1)
axC.set_xlabel("retention fraction"); axC.set_ylabel("cosine vs dense")
axC.set_title("Qwen3-8B M1: summary curve (all layers, all prompts)")
axC.legend(); axC.grid(alpha=0.3)
figC.tight_layout()
pC = PLOT_DIR / "m1_summary_curve.png"
figC.savefig(pC, dpi=150)

plt.show(figA)
for p in (pA, pB, pC):
    print(f"saved: {p}  ({p.stat().st_size/1024:.0f} KB)")
'''

nb = json.loads(NB.read_text(encoding="utf-8"))
target = None
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if "barh" in src and "fig2" in src:
        target = i
        break
assert target is not None, "plot cell not found"

nb["cells"][target]["source"] = NEW_SOURCE.splitlines(keepends=True)
nb["cells"][target]["outputs"] = []          # stale blank-PNG output removed
nb["cells"][target]["execution_count"] = None
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

# verify round-trip
nb2 = json.loads(NB.read_text(encoding="utf-8"))
src = "".join(nb2["cells"][target]["source"])
assert "m1_quality_vs_retention" in src and "m1_summary_curve" in src
print(f"patched cell {target}; new length {len(src)} chars, {src.count(chr(10))+1} lines")
