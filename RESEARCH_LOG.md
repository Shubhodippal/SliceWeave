# SliceWeave — Research Log & Environment Notes

Living record of experiment results, decision-gate verdicts, and the machine
setup needed to reproduce them. Companion to the blueprint (`update.md`) and
`experiments/README.md`. Update after every milestone run.

**Archive policy (2026-09-20):** ALL artifacts (CSV/JSON/plots/traces) are kept
forever, organized per milestone under `results/m*/` (see
`results/ARCHIVE_POLICY.md`); every run appends provenance to
`results/RUN_MANIFEST.md`. Superseded files go to `results/_archive/` with a
timestamp — nothing is ever deleted.

---

## Milestone status (2026-09-20)

### M0 — Exact decomposition ✅ PASSED
- Real Qwen3-4B weights, GPU-resident layer, group sizes 64/128/256
- sum of all micro-experts == dense FFN (rel err ~2e-7); permutation == dense
- 100% retention reproduces baseline logits (cosine ≥ 0.999985)
- Deliverable: `results/csv/exactness_report.csv` (`experiments/00_exact_decomposition.py`)

### M1 — Oracle sparsity ✅ GATE PASSED (STRONG signal)
- Full 3-layer sweep via `notebooks/oracle_milestone1.ipynb`: 306 units,
  27,261 rows, 71 min on the 3060 + RAM-offload setup
- cosine@30% = 0.9966, top-1@30% = 0.94; even 5% retention: 0.9926 / 0.92
- Layer 29 most sensitive; reasoning hardest category (0.9914 @30%)
- §14 verdict: **continue to router**
- Deliverable: `results/csv/oracle_ffn.csv`

### M2 — Task specialization ⚠️ RUN — none found at 128-neuron block level
- `notebooks/task_specialization.ipynb`; 45 (layer, task) captures, 763 tokens/layer
- ALL experts near-universal entropy (H≈1.93/2.0) at every probed layer;
  0% task-selective; universal core 5/76, 0/76, 7/76; heatmaps show no task bands
- POST-HOC FLAW FOUND (M3): M2's per-task-median threshold pinned
  P(active|task) ∈ [0.435, 0.500] **by construction** — degenerate metric;
  pooled per-expert threshold is the correct one (see M3)
- Deliverable: `results/csv/task_specialization.json`

### M3 — Neuron clustering ✅ RUN — verified null; specialization does not emerge
- `notebooks/neuron_clustering.ipynb`: capture cached in `results/traces/m3_capture/`
  (45 matrices, 2,289 tokens), torch k-means per layer, k ∈ {38,76,152,304}
- **Exactness PASSED**: clustered permutation identity holds algebraically
  (verify_reorder_equivalence) and physically (permuted forward cosine 0.999936;
  weights restored, restored cosine 0.999985)
- **Pooled-metric comparison (contiguous vs clustered, k=76):**
  meanH 1.850 vs 1.863; specialized fraction 0% both — clustering does NOT
  create task structure. Granularity sweep k=38..304: 0% everywhere.
  Contribution-metric arm (‖W_down z_e‖, M1's score): 0%.
- Intra-cluster correlation only 0.07–0.15 → activations dominated by heavy-tailed
  neurons, not clean co-activating groups
- **Interpretation:** M1's per-token sparsity is real but **token-local**
  (syntax/position/token-identity), NOT task-conditional at FFN-block level.
  Per §55.7 prompt-level retrieval (M5) is dead at this granularity; token-level
  routing (M4) can still stand on M1 + the M2/M3 nulls as evidence.
- Deliverables: `results/csv/cluster_statistics.csv`,
  `results/csv/cluster_specialization.json`, `results/traces/expert_assignments.npy`

## Decision point (2026-09-20) — choose ONE before M4
1. **Pivot per §55.7**: drop prompt-level routing; M4 = token-level router
   (predict per-token importance from hidden state, no task conditioning).
   Paper story: "dense FFNs are token-locally sparse but not task-partitioned"
   — M2/M3 nulls become evidence, not failures.
2. One more attempt: cluster on contribution vectors or much finer k (608+).
3. Change conditioning variable: position/syntax class instead of task category.

### M3b — Option 2 test ✅ RUN — CONFIRMED NULL (the decisive one)
- `scripts/run_option2.py` (headless twin of `notebooks/option2_granularity.ipynb`
  — the VS Code kernel was hung, so the script form was used; both exist)
- **Arm 2a (finer k = 608/1216/2432 → 16/8/4 neurons per expert):** meanH rises
  slightly (1.86→1.88), specialized stays 0%. Also flag: at k≥1216 there is
  <1 token per expert of calibration data — statistics are starved, so finer
  clustering cannot be tested properly with the current capture budget.
- **Arm 2b (token-space MI — the partition-independent test):** cluster TOKENS
  by their 76-dim contribution fingerprint, measure MI(cluster; task) with
  200-permutation test. Best: MI = 0.998 bits of 3.798 (26%) at k=76, but
  **perm-p = 1.000 across the board** — the clustering is fitting generic
  activation-magnitude structure, not task structure. MI/token-clusters curve
  rises smoothly with k (overfitting signature), identical across layers.
- **VERDICT: three independent rigorous nulls (M2, M3, M3b).** Task-level
  specialization does not exist in Qwen3-4B FFN blocks at any tested
  granularity, metric, or representation. Per §55.7: pivot to token-level
  routing is now mandatory, not optional.
- Deliverables (append-only archive): `results/m3b_granularity_option2/`
  (csv/arm2a_fine_k.csv, csv/arm2b_token_mi.csv, plots/option2_summary.png,
  json/option2_verdict.json) + `results/RUN_MANIFEST.md` entry.

## Decision point — RESOLVED (2026-09-20): pivot to token-level routing (option 1)
- M4 = token router: predict per-token block importance from the hidden state
  alone; benchmark recall vs M1 oracle rankings; §80 overhead budget check.

### M4 — Token-level router ✅ RUN — GATE FAILED (§55); routing line STOPPED
- `notebooks/token_router.ipynb` (+ headless twins `scripts/run_m4_capture.py`,
  `scripts/run_m4_router.py` — kernel hung again; logic identical, kept in sync)
- **Capture:** paired (x_l,t, oracle ‖W_down z_e‖) for 76 contiguous 128-neuron
  blocks; 414 files / 6,207 token rows across layers 7/18/29, one forward per
  prompt with input+down_proj hooks on all 3 layers simultaneously (2.4 min).
  Cache: `results/traces/m4_capture/` (resumable).
- **Split discipline:** BY PROMPT 70/30 (97/41 prompts, seed 0) — never
  token-random (neighbor leakage).
- **Routers (holdout recall@19 of 76):** always-on static frequency 0.52–0.65;
  random 0.25; centroid (§18) 0.37–0.51; kNN (m=16, nonparametric) 0.56–0.68
  (best token-conditioned); linear probe (76×2560, GD) 0.30–0.34 (overfits;
  in-sample ≤0.54). At K=30 only layer 18 grazes 0.73.
- **§55 gate @K=19 (threshold 0.70): FAILED at every layer** (0.56/0.68/0.59).
- **Decisive diagnostic:** static always-on (NO token input) is within ~3–10
  points of the best token-conditioned router; kNN's in-sample ceiling is
  itself only 0.61–0.71. The oracle importance vector is only weakly
  determined by the FFN input x beyond a global frequency prior.
- **§80 overhead: PASS** — router 2.7–4.6% of one dense FFN pass (centroid
  0.196 ms, probe 0.116 ms vs FFN 4.25 ms @ batch 64, resident layer 8).
  Overhead is not the bottleneck; predictability is.
- **Quality proxy caveat:** masked-logit cosine at layer 18 is flat
  (~0.9998 for ALL routers incl. random, top-19 budgets, ~150 slots) —
  middle-layer FFN output barely moves last-token logits; the proxy is not
  discriminating at this depth (consistent with M1's layer-sensitivity: 29
  most sensitive).
- **VERDICT (§55 item 2 — "routing recall too low even with expensive
  predictors"): STOP the routing line. Do NOT proceed to M6 adaptive-K.**
  Four-milestone paper story complete: dense FFNs are token-locally sparse
  (M1) but the sparsity pattern is neither task-partitioned (M2/M3/M3b) nor
  predictable from the incoming hidden state (M4) beyond a static prior.
  Post-hoc fine-grained conditional execution of dense FFN blocks is
  information-theoretically limited at 128-neuron granularity.
- Remaining viable directions: coarser granularity (256/512-neuron blocks,
  where static coverage is higher), importance-from-z (defeats the purpose),
  or pivoting the contribution to placement/prefetch (§56 H6/H7).
- Deliverables: `results/m4_token_router/` (csv/router_vs_oracle.csv,
  plots/router_recall.png, json/router_summary.json) + capture cache +
  `results/RUN_MANIFEST.md` entry.

### M4b — Router × granularity sweep ✅ RUN — NULL CONFIRMED at every granularity
- `notebooks/token_router_granularity.ipynb` (+ headless twin
  `scripts/run_m4b_granularity.py`). Two upgrades over M4:
  1. **Corrected router input**: the MLP's own input (post-attention hidden
     state) instead of M4's pre-attention layer input — the true inference-time
     signal. Helped learned routers slightly; did not change the verdict.
  2. **Exact big-block importance** from cached z: ‖W_big z‖ = √(Σ‖W_child z‖²)
     — one capture serves all granularities. 1024 invalid (9728/1024=9.5).
- Granularity ladder (K = 25% of blocks, aggregated over layers 7/18/29):

  | g | blocks | K | knn | static | margin | z-energy ceiling |
  |---|---|---|---|---|---|---|
  | 128 | 76 | 19 | 0.612 | 0.573 | +0.040 | 0.945 |
  | 256 | 38 | 10 | 0.643 | 0.612 | +0.031 | 0.948 |
  | 512 | 19 | 5 | 0.669 | 0.653 | +0.016 | 0.947 |

- **Both gates fail everywhere**: recall@K < 0.70 at every granularity, and
  the learned-over-static margin SHRINKS as blocks get coarser (+0.040 →
  +0.016) — the static prior absorbs the granularity gain.
- **z-energy diagnostic (the decisive mechanism result):** ranking blocks by
  mean z² AFTER gate/up gives 0.93–0.96 recall at every granularity — the
  importance signal exists in the FFN pipeline but is destroyed for
  prediction purposes by the gate/up product; it is not decodable from x at
  any block size.
- **VERDICT: granularity escape route closed.** Routing line fully dead
  (M4 + M4b). Only remaining skip mechanism is post-hoc z-energy selection
  (saves the down_proj third of FFN FLOPs only — appendix-grade). Paper story
  complete: M1 (sparsity real) + M2/M3/M3b (not task-partitioned) + M4/M4b
  (not predictable from input at any granularity; signal exists only
  post-gate/up). → proceed to paper.
- Deliverables: `results/m4b_granularity_router/` (csv/router_granularity.csv,
  plots/granularity_router_recall.png, json/granularity_router_verdict.json)
  + `results/traces/m4b_capture/` + RUN_MANIFEST entry.

### M4c — z-energy autopsy ✅ RUN — POSITIVE RESULT (the mechanism that survives)
- `notebooks/zenergy_downproj.ipynb` (+ headless twin `scripts/run_m4c_zenergy.py`).
  The one defensible mechanism after M4/M4b: rank 128-neuron blocks by mean z²
  AFTER gate/up, skip low-energy blocks in **down_proj only** (gate/up must
  run fully — ⅔ of FFN FLOPs).
- **Quality (masked forwards, layers 8 + 18, 3 prompts × 11–16 slots × 6
  retentions):** z-energy matches the ORACLE to 4 decimal places at every
  retention (cosine 0.9998 @ 30% on both layers); beats random (0.9994 on
  layer 18). M4b's 0.95 rank agreement now has quality parity to back it.
- **Wall-clock (resident layer 8, bf16, batch 64, CUDA-timed):**
  dense down_proj 1.433 ms → gathered path 0.112 ms @ 30% retention =
  **7.9% of dense time (12.7×)**; 50% retention → 10.8%. Better than the
  FLOP model because the dense GEMM is memory-bandwidth-bound (streams all
  47 MB of weights; gathered streams proportionally less) — §77 answered in
  the favourable direction.
- **Caveats:** same k blocks across batch (row-specific gathers need a
  grouped-GEMM kernel); weight gather pre-applied (deployment needs fused
  gather+GEMM); scope is down_proj only → ≤⅓ of FFN FLOPs (~11% of decoder
  FLOPs). A measured kernel result, not a full-model speedup claim.
- **Place in the paper:** the constructive punchline. "Importance is created
  inside the FFN by the gate/up product and is immediately measurable from z
  at negligible cost — enabling a measured 12.7× down_proj speedup at 30%
  retention with oracle-parity quality — but it cannot be exported backwards
  to a pre-FFN router (M4/M4b)." Negative + positive halves = one mechanism story.
- Deliverables: `results/m4c_zenergy/` (csv/zenergy_quality.csv,
  plots/zenergy_summary.png, json/zenergy_verdict.json) + RUN_MANIFEST entry.

---

## Key engineering facts
- Batched oracle (`oracle_multi_token` + `FFNMaskedForward.set_batch_masks`)
  verified equivalent to sequential masking (cosine 0.999989, bf16 noise),
  ~6–8× speedup, ~14 s/unit. Equivalence proof: `scripts/debug_batched_oracle.py`
- Oracle masks ONLY the analyzed token slot — one token's ranking must never
  blanket the whole prompt (per-token question, §9/§20)
- `FFNMaskedForward` batch protocol (M4 lesson): `set_batch_masks(layer, keeps,
  positions)` expects the INPUT BATCH to be replicated to B=len(positions) rows
  (row i's mask applies at positions[i], `zm[i, t]`) — copy the
  `oracle_multi_token` pattern: `input_ids.expand(B, -1)`, then read row i's
  logits at its own position. A batch-1 forward with a multi-row mask raises
  IndexError. Also: pop the batch entry before any dense forward (stale masks
  crash the baseline), and use `logits_to_keep=1` on batched forwards — full
  vocab logits at B=71 need ~9 GiB and OOM the 3060.
- Masks for `FFNMaskedForward` must be created on CPU: offloaded layers'
  parameters sit on META between forwards, so
  `next(ffn.parameters()).device` returns meta and a meta mask crashes the
  hook (the hook casts masks to z's dtype/device itself).
- torch 2.14 forward pre-hook signature: `hook(mod, args, kwargs)` with
  `with_kwargs=True`; a 3-positional-arg lambda raises TypeError.

## Machine facts (verified 2026-09-18)
- GPU: RTX 3060 Laptop 6 GiB; RAM 29.9 GiB usable; E: ~908 GiB free
- Python 3.13 **regular** build ONLY — the default `py` launcher points to the
  free-threaded 3.13t, which is torch-incompatible; always `py -3.13`
- torch 2.14.0+cu126, transformers 5.17.0 (use `dtype=`, not `torch_dtype=`),
  accelerate 1.15.0, sklearn 1.9.1, scipy 1.18.1
- `HF_HOME=E:\hf_cache` (user env var); Qwen3-4B bf16 (~8 GB) cached there
- Venv: `.venv` → `.venv\Scripts\python.exe`

## Known issue: Smart App Control transiently blocks unsigned wheels
- SAC is ON (`VerifiedAndReputableDesktop` policy). First load of unsigned pyd
  files (sklearn `_heap.pyd`, pandas `tzconversion.pyd`) can be blocked while the
  cloud reputation verdict is pending; retries succeed once the verdict is cached
- Diagnose via Event Log `Microsoft-Windows-CodeIntegrity/Operational`, Id 3077/3076
- `src/_sklearn_shim.py` is a dormant fallback (self-disables when
  `sklearn.metrics` imports cleanly)
- `src/clustering.py` deliberately avoids sklearn (torch k-means; scipy corr
  fallback) to minimize exposure

## Codebase conventions
- Prompt bank `src/datasets.py`: ~498 prompts, 15 categories (coding = 201 across
  10 language/topic banks; 8 practical categories × 20; general categories 18–25
  each). Helpers: `all_prompts()`, `sample_prompts(n, seed)`,
  `generated_arithmetic(n)`, `bank_stats()`
- Typed accessors in `model_utils`: `get_mlp(model, layer_key)`,
  `mlp_intermediate_size()`, `mlp_down_weight(model, key, device=...)` (device
  pinning for offloaded layers), `pick_resident_layer(model)` for weight-math on
  GPU-resident layers (accelerate offload leaves meta params between forwards;
  resolve via `AlignDevicesHook.weights_map`)
- `layer_key = "model.layers.N"`; FFN capture key = `f"{layer_key}.down_proj"`

## Notebook conventions (established pattern)
- One notebook per milestone; SMALL cells, one job each; a markdown note between
  every pair of cells explaining "what just happened / what happens next"; first
  cell is a numbered process-flow diagram; never a master mega-cell
- Scripts in `experiments/` remain the paper artifacts; notebooks are the cockpit

## Next steps
1. ~~DECIDE between the three options~~ RESOLVED (option 1) → M4 executed →
   **GATE FAILED, routing line STOPPED** (see M4 verdict above)
2. ~~Coarser-granularity retest~~ DONE → M4b → **NULL CONFIRMED at every
   granularity** (see M4b verdict above) — routing line fully closed
3. ~~z-energy autopsy~~ DONE → M4c → **POSITIVE: 12.7× down_proj wall-clock
   at 30% retention, oracle-parity quality** (see M4c verdict above)
4. **Proceed to the paper** with the completed arc: M1 (sparsity real) →
   M2/M3/M3b (not task-partitioned) → M4/M4b (not predictable from input at
   any granularity) → M4c (but measurable post-hoc from z: 12.7× down_proj
   measured speedup, oracle-parity quality). Framing: "dense FFNs are
   token-locally sparse; exploitability requires measuring z, not predicting
   from x."

## Qwen3-8B study (2026-09-21) - M0 PASS
- Fresh pinned download (b968826d) after silent-stall salvage attempt was discarded; HF_HUB_DISABLE_SYMLINKS=1 required on this host.
- M0 exact decomposition PASSED on the first executed run (20260921T044250Z): identities exact at all group sizes (rel err ~2e-7), full-retention cosine >= 0.99999, top-1 match.
- 8B geometry: 36 layers, hidden 4096, intermediate 12288 (96 blocks @128). Resident layer model.layers.2; 5 GPU / 35 CPU layers, no disk offload.
- Next: M1 oracle sparsity sweep in notebooks/qwen3_model_study.ipynb.

## Qwen3-8B study (2026-09-21) - M1 GATE PASSED (STRONG)
- oracle_milestone1.ipynb parameterized with MODEL_KEY switch (4B shared cache vs 8B pinned bundle); per-milestone notebook convention restored - the study workbench stays M0-only.
- 306 units / 23,931 rows / 202.7 min on the 3060+RAM-offload setup. cosine@30%=0.9967, top1@30%=0.94; 5% retention still 0.9939/0.92 - matches the 4B pattern closely.
- Layer sensitivity: 29 most sensitive (0.9913@30%), 18 least (0.9956). Hardest categories: reasoning, coding.
- Next: M2 task-structure probes in their own notebook against the 8B bundle.
