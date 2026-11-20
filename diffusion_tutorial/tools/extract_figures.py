"""Dump every image output of an executed notebook to PNG files (for reviewing).

    python tools/extract_figures.py 02_ddpm_from_scratch.ipynb /tmp/figs02
Also prints each cell's text outputs (stdout / errors) in order, with the cell index.
"""
import base64
import json
import sys
from pathlib import Path

nb_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
nb = json.loads(nb_path.read_text())
n_img = 0
for ci, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    for oi, out in enumerate(cell.get("outputs", [])):
        if out.get("output_type") == "stream":
            txt = "".join(out["text"])
            print(f"--- cell {ci} [{out['name']}]\n{txt[-2000:]}")
        elif out.get("output_type") == "error":
            print(f"--- cell {ci} ERROR {out['ename']}: {out['evalue']}")
        data = out.get("data", {})
        if "text/plain" in data and "image/png" not in data:
            print(f"--- cell {ci} [result]\n{''.join(data['text/plain'])[-1000:]}")
        if "image/png" in data:
            p = out_dir / f"cell{ci:03d}_{oi}.png"
            p.write_bytes(base64.b64decode(data["image/png"]))
            n_img += 1
            print(f"--- cell {ci} [image] -> {p}")
print(f"\n{n_img} images written to {out_dir}")
