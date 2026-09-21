# runtime/ — future optimized local runtime (§71, Milestone 11)

This directory is intentionally a stub. Blueprint §71 is explicit:

> Do not start by modifying llama.cpp. Prove the scientific assumptions first.

Planned contents once Milestones 0-10 succeed:

- `cpp/` — custom sparse block kernels (grouped GEMM over contiguous experts, §78)
- `cuda/`/triton — fused gather+GEMM for expert blocks (§79)
- `gguf/` — micro-expert container format with per-block metadata (§44-§45)
- `scheduler/` — productionized version of `src/scheduler.py` with async
  prefetch (§26) and real measured cost model (§65)
