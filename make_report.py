import datetime
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import config
import gslib.out
from gslib.stats import load_csv

A4 = (8.27, 11.69)

SECTIONS = [
    ("A", "A. Where the bytes go", "A_storage", "A_storage",
     "Every Gaussian carries 62 float32 values. Grouping them by job shows which part is eating the file, "
     "which is exactly where compression should start."),
    ("B", "B. Opacity: who is actually visible?", "B_opacity", "B_opacity",
     "Opacity is stored as a logit and goes through a sigmoid to land in (0, 1). Gaussians that are almost fully "
     "transparent barely touch the final image, so they are first in line for pruning (Compact3DGS, "
     "Papantonakis et al.)."),
    ("C", "C. Shape check: needles, discs and blobs", "C_scale", "C_scale",
     "Scale is stored in log space; exp gives the three ellipsoid axes, sorted short < mid < long. A Gaussian "
     "counts as a needle when mid < 0.1 x long, a disc when short < 0.1 x mid, and a 3D blob otherwise."),
    ("D", "D. Spherical Harmonics: is the extra color worth it?", "D_sh", "D_sh",
     "View-dependent color lives in SH coefficients of degree 0 to 3, 48 numbers in total. When a Gaussian's "
     "higher-degree coefficients are basically zero, storing them is pure overhead. This is the idea behind "
     "adaptive SH (Papantonakis et al.)."),
    ("D2", "D2. How many SH degrees do we really need?", None, "D2_sh_threshold",
     "A Gaussian 'needs' degree l when the RMS of its degree-l coefficients clears the threshold. This table "
     "tests how much the answer moves when the threshold moves."),
    ("E", "E. Where the Gaussians live", "E_space", "E_space",
     "Gaussian density projected onto the three coordinate planes. Dense spots mark where the detail is; the "
     "faint scatter far out is floaters and background."),
    ("E2", "E2. How crowded is the hotspot?", None, "E2_concentration",
     "We drop a sphere on the densest region and compare two numbers: the share of Gaussians inside it and the "
     "share of scene volume it takes up."),
    ("F", "F. Precision diet: float16 and 8-bit", None, "F_quant",
     "Each attribute group is squeezed into float16 or 8-bit, decoded again and compared with the original. "
     "Error is measured against the group's standard deviation: the lower it is, the better the group handles "
     "quantization."),
    ("F2", "F2. Position precision vs. Gaussian size", None, "F2_position",
     "For positions, standard deviation is the wrong ruler. What matters is the error compared with the "
     "Gaussian's own size: move a splat further than its own width and the geometry falls apart."),
    ("G", "G. Glow-up check: 7k vs 30k iterations", "G_compare", "G_compare",
     "Same scene, two checkpoints. This is what densification does to the Gaussian count, the file size and "
     "the shapes along the way."),
    ("H", "H. Quick wins: what if we just...", "H_whatif", "H_whatif",
     "Estimated file size after a few no-brainer moves: drop normals, cut SH degrees, switch to float16, prune "
     "by opacity. This covers size only; image quality still has to be checked with renders and PSNR."),
]

LAYOUT_TEXT = [
    ["x y z", "3", "Center position", "as is"],
    ["nx ny nz", "3", "Normal (always 0)", "skip it"],
    ["f_dc_0..2", "3", "Base color (SH degree 0)", "RGB = 0.5 + 0.282 * f_dc"],
    ["f_rest_0..44", "45", "SH degrees 1-3, laid out R..G..B", "used at render time"],
    ["opacity", "1", "Opacity (logit)", "sigmoid"],
    ["scale_0..2", "3", "Axis lengths (log)", "exp"],
    ["rot_0..3", "4", "Quaternion w, x, y, z", "normalize"],
]


def wrap(s, width):
    return "\n".join(textwrap.wrap(str(s), width)) if s != "" else ""


def draw_table(ax, rows_or_header, cells=None, fontsize=9):
    ax.axis("off")
    if cells is None:
        header = list(rows_or_header[0].keys())
        cells = [[r[h] for h in header] for r in rows_or_header]
    else:
        header = rows_or_header
    w = max(12, int(95 / len(header)))
    t = ax.table(cellText=[[wrap(c, w) for c in row] for row in cells],
                 colLabels=[wrap(h, w) for h in header], loc="upper center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(fontsize)
    t.scale(1, 1.5)
    cells_d = t.get_celld()
    lines = {}
    for (r, _), cell in cells_d.items():
        lines[r] = max(lines.get(r, 1), cell.get_text().get_text().count("\n") + 1)
    for (r, _), cell in cells_d.items():
        cell.set_height(cell.get_height() * (1 + 0.75 * (lines[r] - 1)))
    first_col = {r: c for (r, c2), c in cells_d.items() if c2 == 0}
    total = sum(c.get_height() for c in first_col.values())
    if total > 1:
        for cell in cells_d.values():
            cell.set_height(cell.get_height() / total)
        t.set_fontsize(fontsize / min(total, 1.4))
    for (r, _), cell in cells_d.items():
        if r == 0:
            cell.set_facecolor("#DDE6F0")
            cell.set_text_props(weight="bold")
    return t


def text_block(fig, y, lines, size=12, width=None, **kw):
    width = width or int(1080 / size)
    for line in lines:
        for sub in textwrap.wrap(line, width) or [""]:
            fig.text(0.07, y, sub, fontsize=size, va="top", **kw)
            y -= 0.018 * size / 10
        y -= 0.006
    return y


def main():
    findings = {}
    fpath = os.path.join(config.OUT_DIR, "findings.json")
    if os.path.exists(fpath):
        with open(fpath, encoding="utf-8") as f:
            findings = json.load(f)

    pdf_path = os.path.join(config.OUT_DIR, "report.pdf")
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=A4)
        fig.text(0.07, 0.94, "3D Gaussian Splatting Output Analysis", fontsize=20, weight="bold", va="top")
        fig.text(0.07, 0.905, f"Scene: {config.SCENE}    |    Created: {datetime.date.today():%d/%m/%Y}",
                 fontsize=12, va="top", color="#555")
        y = text_block(fig, 0.87, ["At a glance"], size=14, weight="bold")
        ov = os.path.join(config.TAB_DIR, "overview.csv")
        if os.path.exists(ov):
            ax = fig.add_axes([0.07, y - 0.33, 0.86, 0.32])
            draw_table(ax, load_csv(ov), fontsize=10)
            y -= 0.35
        y = text_block(fig, y, ["- " + l for l in findings.get("overview", [])])
        y = text_block(fig, y - 0.01, ["Anatomy of one Gaussian (.ply layout)"], size=14, weight="bold")
        ax = fig.add_axes([0.07, y - 0.22, 0.86, 0.21])
        draw_table(ax, ["Property", "Floats", "What it is", "How to decode"], LAYOUT_TEXT, fontsize=10)
        y -= 0.25
        y = text_block(fig, y, ["Contents: " + "  |  ".join(s[1] for s in SECTIONS)], size=11, color="#555")
        pdf.savefig(fig)
        plt.close(fig)

        for key, title, fig_name, tab_name, desc in SECTIONS:
            img = os.path.join(config.FIG_DIR, f"{fig_name}.png") if fig_name else None
            tab = os.path.join(config.TAB_DIR, f"{tab_name}.csv")
            if not (img and os.path.exists(img)) and not os.path.exists(tab):
                print(f"  skipping section {key} (no data yet)")
                continue
            fig = plt.figure(figsize=A4)
            fig.text(0.07, 0.95, title, fontsize=16, weight="bold", va="top")
            y = text_block(fig, 0.915, [desc], size=12, color="#333")
            if findings.get(key):
                y = text_block(fig, y - 0.005, ["TL;DR"], size=13, weight="bold")
                y = text_block(fig, y, ["- " + l for l in findings[key]], size=12)
            if img and os.path.exists(img):
                im = plt.imread(img)
                h = min(0.42, 0.86 * im.shape[0] / im.shape[1] * A4[0] / A4[1])
                ax = fig.add_axes([0.07, y - h - 0.01, 0.86, h])
                ax.imshow(im)
                ax.axis("off")
                y -= h + 0.03
            if os.path.exists(tab):
                ax = fig.add_axes([0.05, 0.04, 0.90, y - 0.06])
                draw_table(ax, load_csv(tab))
            pdf.savefig(fig)
            plt.close(fig)
    print("Created", pdf_path)


if __name__ == "__main__":
    main()
