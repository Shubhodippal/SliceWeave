"""One-off patch 2: logits_to_keep=1 in the quality-proxy cell (OOM fix)."""
import json
from pathlib import Path

p = Path(r"e:\Project\SliceWeave\notebooks\token_router.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))
cell = nb["cells"][-1]
src = "".join(cell["source"])

src = src.replace(
    "out = model(**batch_enc).logits[0, -1].float().cpu()",
    "out = model(**batch_enc, logits_to_keep=1).logits[0, -1].float().cpu()")
src = src.replace(
    "base = model(**enc).logits[0, -1].float().cpu()",
    "base = model(**enc, logits_to_keep=1).logits[0, -1].float().cpu()")

assert "logits_to_keep=1" in src
cell["source"] = src.splitlines(keepends=True)
p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("patched: logits_to_keep=1 on both forwards")
