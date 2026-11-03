import numpy as np
import matplotlib.pyplot as plt

import config
from gslib.load import load_gaussians
from gslib import stats as S
from gslib.out import save_fig, save_table, add_findings

raw, g = load_gaussians(config.PLY_MAIN)
print(f"Loaded {g['N']:,} Gaussians from {config.PLY_MAIN}")

print("\n[A] Storage by attribute group")
rows = S.storage_breakdown(raw)
save_table(rows, "A_storage")
parts = rows[:-1]
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.barh([r["Group"] for r in parts][::-1], [r["% of size"] for r in parts][::-1], color="#4C78A8")
for i, r in enumerate(parts[::-1]):
    ax.text(r["% of size"] + 0.5, i, f"{r['% of size']}%  ({r['Total (MB)']} MB)", va="center", fontsize=8)
ax.set_xlabel("% of file size")
ax.set_xlim(0, max(r["% of size"] for r in parts) * 1.35)
ax.set_title("A. Where the bytes go")
save_fig(fig, "A_storage")
sh_high = sum(r["% of size"] for r in parts if r["Group"].startswith("SH degree"))
add_findings("A", [
    f"SH degrees 1-3 (view-dependent color) eat {sh_high:.1f}% of the file on their own. Biggest target by far.",
    "Normals are always 0 yet still take up space. Pure dead weight: dropping them loses nothing.",
])

print("\n[B] Opacity")
op = g["opacity"]
save_table(S.opacity_table(g), "B_opacity")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.5))
a1.hist(op, bins=100, range=(0, 1), color="#F58518")
a1.set_yscale("log")
a1.set_xlabel("opacity (after sigmoid)")
a1.set_ylabel("Gaussians (log)")
a1.set_title("Opacity distribution")
xs = np.sort(op)
a2.plot(xs, np.linspace(0, 100, xs.size), color="#F58518")
for t in (0.005, 0.05):
    a2.axvline(t, ls="--", c="gray")
    a2.text(t, 50, f" {t}", fontsize=8)
a2.set_xscale("log")
a2.set_xlabel("opacity (log)")
a2.set_ylabel("% Gaussians with lower opacity")
a2.set_title("Cumulative distribution")
save_fig(fig, "B_opacity")
add_findings("B", [
    f"Average opacity sits at {op.mean():.3f}, median at {np.median(op):.3f}.",
    f"{S.pct(op < 0.05):.1f}% of Gaussians are below 0.05, which is basically see-through. Prime pruning candidates.",
    f"{S.pct(op > 0.9):.1f}% are close to solid (> 0.9).",
])

print("\n[C] Size and flatness")
f = S.scale_features(g)
save_table(S.scale_table(g), "C_scale")
fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
axs[0].hist(np.log10(f["s_max"]), bins=100, color="#54A24B")
axs[0].set_xlabel("log10(longest axis)")
axs[0].set_title("Size")
axs[1].hist(np.log10(f["aniso"]), bins=100, color="#54A24B")
axs[1].set_xlabel("log10(long / short)")
axs[1].set_title("Elongation")
axs[2].hist(f["flat"], bins=100, range=(0, 1), color="#54A24B")
axs[2].axvline(0.1, ls="--", c="gray")
axs[2].set_xlabel("short / mid  (small = disc-like)")
axs[2].set_title("Flatness")
for a in axs:
    a.set_ylabel("Gaussians")
save_fig(fig, "C_scale")
add_findings("C", [
    f"Typical size: the longest axis has a median of {np.median(f['s_max']):.3g} and a P99 of "
    f"{np.percentile(f['s_max'], 99):.3g} scene units.",
    f"Shape split: {S.pct(f['needle']):.1f}% needles, {S.pct(f['disc']):.1f}% discs hugging surfaces "
    f"(same idea as RTG-SLAM), {S.pct(f['blob']):.1f}% 3D blobs.",
    f"The median long/short ratio is {np.median(f['aniso']):.0f}x. Round Gaussians are rare: just "
    f"{S.pct(f['aniso'] < 2):.1f}% have a ratio under 2.",
])

print("\n[D] SH energy by degree")
rms = S.sh_band_rms(g)
rows = S.sh_table(g)
save_table(rows, "D_sh")
need = S.sh_needed_degree(g)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5))
for b, v in rms.items():
    a1.hist(np.log10(v + 1e-6), bins=100, histtype="step", lw=1.5, label=f"degree {b}")
a1.axvline(np.log10(S.SH_EPS), ls="--", c="gray")
a1.set_xlabel("log10(coefficient RMS)")
a1.set_ylabel("Gaussians")
a1.set_title("SH coefficient magnitude by degree")
a1.legend()
counts = [S.pct(need == d) for d in range(4)]
a2.bar([f"degree {d}" for d in range(4)], counts, color="#B279A2")
for i, c in enumerate(counts):
    a2.text(i, c + 1, f"{c:.1f}%", ha="center", fontsize=8)
a2.set_ylabel("% Gaussians")
a2.set_title(f"Minimum SH degree needed (RMS threshold {S.SH_EPS})")
save_fig(fig, "D_sh")
thr = S.sh_threshold_table(g)
save_table(thr, "D2_sh_threshold")
add_findings("D", [
    f"Base color (degree 0) carries {rows[0]['% color energy']}% of the color energy with just 3 of the 48 coefficients.",
    f"{S.sh_zero_pct(g):.1f}% of Gaussians have all 45 higher-order coefficients at exactly 0 "
    "(the spike at log10 = -6 in the plot). Storing 45 zeros per splat is wasted space.",
])
add_findings("D2", [
    f"The share of Gaussians that only need base color swings from {thr[0]['% need only degree 0']}% "
    f"(threshold {thr[0]['RMS threshold']}) to {thr[-1]['% need only degree 0']}% "
    f"(threshold {thr[-1]['RMS threshold']}). The threshold changes the story a lot.",
    "So the threshold can't be picked by vibes. It has to come from rendering and PSNR, the way "
    "Papantonakis et al. tune it on color error across views.",
])
