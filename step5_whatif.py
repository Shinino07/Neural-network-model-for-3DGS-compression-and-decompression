import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import config
from gslib.load import load_gaussians, save_ply
from gslib import stats as S
from gslib.out import save_fig, save_table, add_findings

raw, g = load_gaussians(config.PLY_MAIN)
print(f"Loaded {g['N']:,} Gaussians from {config.PLY_MAIN}")

print("\n[H] Rough compression")
rows = S.whatif_table(g)
save_table(rows, "H_whatif")
fig, ax = plt.subplots(figsize=(8, 4.5))
labels = [r["Scenario"] for r in rows][::-1]
vals = [r["Size (MB)"] for r in rows][::-1]
ax.barh(labels, vals, color="#E45756")
for i, r in enumerate(rows[::-1]):
    ax.text(r["Size (MB)"], i, f"  {r['Size (MB)']} MB (x{r['Compression (x)']})", va="center", fontsize=8)
ax.set_xlim(0, max(vals) * 1.3)
ax.set_xlabel("MB")
ax.set_title("H. How small can it get?")
save_fig(fig, "H_whatif")
best = rows[-1]
add_findings("H", [
    f"No-brainer moves alone (scenario 9) bring the file down to {best['Size (MB)']} MB, a x{best['Compression (x)']} cut.",
    "For reference, Compact3DGS and Papantonakis et al. reach x15-x30 with codebooks, hash grids and entropy coding.",
    "What each scenario does to image quality stays unknown until we render and measure PSNR.",
])

if "--write" in sys.argv:
    out = os.path.join("data", "variants")
    os.makedirs(out, exist_ok=True)
    op = g["opacity"]
    save_ply(os.path.join(out, "prune_op005.ply"), raw[op >= 0.05])
    sh0 = raw.copy()
    for k in sh0.dtype.names:
        if k.startswith("f_rest_"):
            sh0[k] = 0
    save_ply(os.path.join(out, "sh_deg0.ply"), sh0)
    print(f"  wrote variants to {out}/ (original format kept so viewers can open them)")
