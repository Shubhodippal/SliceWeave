"""Windows-safe sklearn shim.

transformers 5.x imports `sklearn.metrics.roc_curve` at module import time
(generation/candidate_generator.py). On some locked-down Windows machines a
Windows Application Control policy blocks loading sklearn's Cython `_heap`
extension inside `sklearn.metrics._pairwise_distances_reduction`, which breaks
ANY import of sklearn.metrics — and therefore transformers itself — even
though torch and the rest of the stack are fine.

If `sklearn.metrics` imports cleanly, this module is a no-op. Otherwise it
installs a minimal pure-python `roc_curve` into sys.modules so transformers
can import. `roc_curve` is only used by transformers' assisted-decoding
candidate generator — none of this repo's research paths call it.

Import this BEFORE transformers (src.model_utils does it for you).
"""
from __future__ import annotations

import importlib
import sys
import types


def _install_shim() -> bool:
    try:
        importlib.import_module("sklearn.metrics")
        return False  # environment is fine, no shim needed
    except ImportError:
        pass

    mod = types.ModuleType("sklearn.metrics")
    mod.__dict__["__shim__"] = True  # marker, kept out of attribute typing

    def roc_curve(y_true, y_score, *args, **kwargs):
        raise NotImplementedError(
            "sklearn.metrics is unavailable on this machine (Application "
            "Control policy blocked its compiled extensions). roc_curve is "
            "only needed for transformers' assisted decoding, which "
            "SliceWeave experiments do not use.")

    # setattr() instead of `mod.roc_curve = ...`: type checkers reject direct
    # attribute assignment on ModuleType instances
    setattr(mod, "roc_curve", roc_curve)
    sys.modules["sklearn.metrics"] = mod
    return True


SHIM_INSTALLED = _install_shim()
