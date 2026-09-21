# SliceWeave / ParamStream — experiment drivers (§72, §73)

Numbered drivers, one per milestone. Run in order; each is a falsification gate.

| # | Milestone (§73) | Script | Deliverable |
|---|---|---|---|
| 00 | M0 Exactness | `00_exact_decomposition.py` | `results/csv/exactness_report.csv` |
| 01 | M1 Oracle sparsity (one layer) | `01_oracle_ffn.py` | `results/csv/oracle_ffn.csv` |
| 02 | M1 Oracle sweep (all layers) | `02_oracle_all_layers.py` | `results/csv/layer_sensitivity.csv`, `results/plots/layer_sensitivity.png` |
| 03 | M2 Task specialization | `03_task_specialization.py` | `results/csv/expert_task_statistics.csv`, `results/plots/task_expert_heatmap.png` |
| 04 | M3 Neuron clustering | `04_cluster_neurons.py` | `results/csv/cluster_statistics.csv`, `results/traces/expert_assignments.npy` |
| 05 | M4 Token router | `05_centroid_router.py` | `results/csv/router_vs_oracle.csv` |
| 06 | M5 Prompt retriever | `06_prompt_retrieval.py` | `results/csv/candidate_pool_results.csv` |
| 07 | M6 Progressive activation | `07_adaptive_k.py` | `results/csv/adaptive_compute_results.csv` |
| 08 | M6 Quality guard sweep | `08_quality_guard.py` | `results/csv/quality_vs_APR.csv` |
| 09 | M7 Memory trace + cache | `09_memory_trace.py` | `results/traces/expert_trace.csv`, `results/csv/cache_simulation.csv` |
| 10 | M8 Predictive prefetch | `10_prefetch_simulation.py` | `results/csv/prefetch_accuracy.csv`, `results/csv/traffic_reduction.csv` |
| 11 | M9 CPU/GPU scheduler | `11_placement_benchmark.py` | `results/csv/placement_benchmark.csv` |
| 12 | Generation agreement | `12_full_generation.py` | `results/csv/generation_agreement.csv` |
| 13 | Wall-clock check | `13_wall_clock_runtime.py` | `results/csv/wall_clock.csv` |
| 14 | Stage 2/3 scaling | `14_scale_model.py` | `results/csv/scale_model.csv` |

Decision gates (§14/§55/§56):

- M1: 20–30% experts preserve logits → continue; >60–70% → weak signal.
- M4: centroid router recall close to oracle → continue.
- M8: prefetch precision high AND wasted bytes low → keep predictor.
- M13: sparse wall-clock ≥ dense → real; else focus on layout (§90).

Typical first pass on the laptop (Stage 1, Qwen3-4B):

```powershell
python experiments/00_exact_decomposition.py
python experiments/01_oracle_ffn.py --layers 3 --prompts 8
python experiments/02_oracle_all_layers.py --fractions 0.5 0.3 0.2 0.1
python experiments/03_task_specialization.py --layers 4
python experiments/05_centroid_router.py --layers 3
```
