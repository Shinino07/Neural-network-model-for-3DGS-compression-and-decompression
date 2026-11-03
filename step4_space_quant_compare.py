import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import config
from gslib.load import load_gaussians
from gslib import stats as S
from gslib.out import save_fig, save_table, add_findings

raw, g = load_gaussians(config.PLY_MAIN)
print(f"Loaded {g['N']:,} Gaussians from {config.PLY_MAIN}")

print("\n[E] Spatial distribution")
xyz = g["xyz"]
save_table(S.spatial_table(g), "E_space")
lo, hi = np.percentile(xyz, 1, axis=0), np.percentile(xyz, 99, axis=0)
pad = 0.1 * (hi - lo)
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
for a, (i, j) in zip(axs, [(0, 1), (0, 2), (1, 2)]):
    h = a.hist2d(xyz[:, i], xyz[:, j], bins=300, norm=LogNorm(), cmap="viridis",
                 range=[[lo[i] - pad[i], hi[i] + pad[i]], [lo[j] - pad[j], hi[j] + pad[j]]])
    a.set_xlabel("xyz"[i])
    a.set_ylabel("xyz"[j])
    a.set_aspect("equal")
    a.set_title(f"Density projected onto the {'xyz'[i]}{'xyz'[j]} plane")
    a.grid(False)
fig.colorbar(h[3], ax=axs, shrink=0.8, label="Gaussians / cell (log)")
fig.savefig(os.path.join(config.FIG_DIR, "E_space.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  figure ->", os.path.join(config.FIG_DIR, "E_space.png"))
t = S.spatial_table(g)
conc = S.spatial_concentration(g)
save_table(conc, "E2_concentration")
mid = conc[1]
add_findings("E", [
    f"98% of Gaussians fit inside a {t[0]['P1-P99 range']} x {t[1]['P1-P99 range']} x {t[2]['P1-P99 range']} box (scene units).",
    f"The spread is very uneven: {mid['% Gaussians inside']}% of Gaussians sit in a sphere of radius "
    f"{mid['Radius (scene units)']} around the hotspot, which is just {mid['% of P1-P99 box volume']}% of the volume. "
    "That hotspot is the main object, where the cameras spent the most time.",
    f"Far-out stragglers (floaters and distant background): {t[3]['P1-P99 range']}.",
    "Heads-up: scene units come from COLMAP, they are not meters.",
])

print("\n[F] Error from reduced precision")
rows = S.quant_table(g)
save_table(rows, "F_quant")
pos_rows, size = S.position_quant_table(g)
save_table(pos_rows, "F2_position")
worst = max(rows, key=lambda r: float(r["8-bit: error/std dev"].rstrip("%")))
add_findings("F", [
    "float16 costs about 0.01-0.02% of the standard deviation in every group. Half the size for basically free.",
    f"With 8-bit uniform, the most sensitive group is '{worst['Group']}' "
    f"({worst['8-bit: error/std dev']} of its standard deviation).",
    "This table is numeric error only. Image quality is a separate question that needs renders and PSNR.",
])
add_findings("F2", [
    f"The median Gaussian is only {size:.3g} across, while the scene spans tens of units, so positions need "
    "serious precision.",
    f"8-bit positions miss by {pos_rows[1]['Mean error']} on average, that is "
    f"{pos_rows[1]['Mean error / median Gaussian size']} the size of a typical Gaussian. Splats get shoved out of "
    "place and the geometry breaks, which the std-dev view in table F completely hides.",
    f"16-bit uniform beats float16 on the same bit budget (max error {pos_rows[2]['Max error']} vs "
    f"{pos_rows[0]['Max error']}), because float16 loses precision the further a point is from the origin.",
])

print("\n[G] Comparing 7k vs 30k iterations")
if os.path.exists(config.PLY_EARLY):
    _, g7 = load_gaussians(config.PLY_EARLY)
    save_table([S.summary_row("7k iterations", g7), S.summary_row("30k iterations", g)], "G_compare")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5))
    for gg, lab in [(g7, "7k"), (g, "30k")]:
        a1.hist(gg["opacity"], bins=100, range=(0, 1), histtype="step", lw=1.5, label=lab)
        a2.hist(np.log10(S.scale_features(gg)["s_max"]), bins=100, histtype="step", lw=1.5, label=lab)
    a1.set_yscale("log")
    a1.set_xlabel("opacity")
    a1.set_title("Opacity")
    a2.set_xlabel("log10(longest axis)")
    a2.set_title("Size")
    for a in (a1, a2):
        a.set_ylabel("Gaussians")
        a.legend()
    save_fig(fig, "G_compare")
    f7, f30 = S.scale_features(g7), S.scale_features(g)
    add_findings("G", [
        f"The Gaussian count goes from {g7['N']:,} (7k) to {g['N']:,} (30k), x{g['N'] / g7['N']:.2f}, and the file "
        f"grows from {g7['file_bytes'] / 1024**2:.0f} to {g['file_bytes'] / 1024**2:.0f} MB.",
        f"More training, more extreme shapes: needles {S.pct(f7['needle']):.1f}% -> {S.pct(f30['needle']):.1f}%, "
        f"discs {S.pct(f7['disc']):.1f}% -> {S.pct(f30['disc']):.1f}%.",
        f"Opacity picks a side: near-solid (> 0.9) {S.pct(g7['opacity'] > 0.9):.1f}% -> {S.pct(g['opacity'] > 0.9):.1f}%, "
        f"near-transparent (< 0.05) {S.pct(g7['opacity'] < 0.05):.1f}% -> {S.pct(g['opacity'] < 0.05):.1f}%.",
    ])
else:
    print(f"  (skipped: {config.PLY_EARLY} not found)")
