"""Render Mermaid .mmd → high-res PNG via mermaid.ink."""

import base64, zlib, json, urllib.request, sys
from pathlib import Path

mmd_path = Path("figures/full_pipeline_flow.mmd")
out_png  = Path("figures/full_pipeline_flow.png")
out_svg  = Path("figures/full_pipeline_flow.svg")

mmd_text = mmd_path.read_text(encoding="utf-8")
if mmd_text.startswith("---"):
    end = mmd_text.find("---", 3)
    if end != -1:
        mmd_text = mmd_text[end+3:].strip()

state = json.dumps({
    "code": mmd_text,
    "mermaid": {"theme": "default"},
    "autoSync": True,
    "updateDiagram": True,
})

compressed = zlib.compress(state.encode("utf-8"), level=9)
encoded = base64.urlsafe_b64encode(compressed).decode("ascii")

# SVG (vector, lossless)
for fmt, out_path, params in [
    ("svg", out_svg, ""),
    ("img", out_png, "?type=png&width=4000&height=2000&bgColor=F8FAFC"),
]:
    url = f"https://mermaid.ink/{fmt}/pako:{encoded}{params}"
    print(f"Fetching {fmt}... (URL len={len(url)})")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        out_path.write_bytes(data)
        print(f"  Saved: {out_path} ({len(data)//1024}KB)")
    except Exception as e:
        print(f"  Failed: {e}", file=sys.stderr)

print("Done.")
