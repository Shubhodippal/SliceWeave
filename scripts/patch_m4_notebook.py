"""One-off patch: clear stale batch masks before the dense baseline forward in
notebooks/token_router.ipynb (quality-proxy cell)."""
import json
from pathlib import Path

p = Path(r"e:\Project\SliceWeave\notebooks\token_router.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))

TARGET = "base = model(**enc).logits[0, -1].float().cpu()"
PATCH_LINE = "        masked.batch_masks.pop(QLK, None)   # dense baseline: NO stale masks\n"

patched = 0
for cell in nb["cells"]:
    src = cell.get("source", [])
    if isinstance(src, str):
        lines = src.splitlines(keepends=True)
    else:
        lines = list(src)
    text = "".join(lines)
    if TARGET in text and "batch_masks.pop" not in text:
        for i, ln in enumerate(lines):
            if ln.rstrip("\n") == "        with torch.no_grad():" and \
               i + 1 < len(lines) and TARGET in lines[i + 1]:
                lines.insert(i, PATCH_LINE)
                patched += 1
                break
    cell["source"] = lines

p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"patched {patched} occurrence(s)")
