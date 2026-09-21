"""SliceWeave / ParamStream core library (update.md blueprint).

Layers:
    model_utils   - loading, VRAM+RAM budgeting, FFN discovery
    ffns          - FFN abstraction + exact micro-expert decomposition
    micro_experts - partitioning, importance scores, exactness checks
    hooks         - activation capture and masked/replacement forward
    oracle        - Phase 1 oracle sparsity machinery
    clustering    - Phase 3 neuron clustering + reordering
    token_router  - Phase 4 centroid router + router-vs-oracle recall
    quality_guard - Phase 6 progressive activation + residual estimator
    prompt_router - Phase 5 prompt-level candidate pool retrieval
    cache         - expert trace + LRU/LFU cache simulation
    prefetch      - next-use / cross-layer prefetch predictor simulation
    scheduler     - tier placement + CPU/GPU cost-model scheduler
    metrics       - logit quality + APR/CPR/MTR/recovery metrics
    datasets      - multi-category calibration/eval prompts
"""
from .model_utils import (  # noqa: F401
    describe_model,
    ffn_layer_names,
    find_ffn_modules,
    load_model_and_tokenizer,
)
from .metrics import compute_all  # noqa: F401
