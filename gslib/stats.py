import csv
import os
import numpy as np

from .load import SH_BANDS, property_groups

MB = 1024 ** 2


def pct(mask):
    return 100.0 * np.count_nonzero(mask) / mask.size


def save_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def load_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def storage_breakdown(raw):
    names = raw.dtype.names
    n = len(raw)
    total = raw.dtype.itemsize
    rows = []
    for grp, props in property_groups(names).items():
        b = sum(raw.dtype[p].itemsize for p in props)
        rows.append({"Group": grp, "Floats": len(props), "Bytes/Gaussian": b,
                     "Total (MB)": round(b * n / MB, 2), "% of size": round(100 * b / total, 1)})
    rows.append({"Group": "TOTAL", "Floats": len(names), "Bytes/Gaussian": total,
                 "Total (MB)": round(total * n / MB, 2), "% of size": 100.0})
    return rows


OPACITY_THRESHOLDS = [0.005, 0.01, 0.05, 0.1, 0.5]


def opacity_table(g):
    op = g["opacity"]
    rows = [{"Opacity threshold": f"< {t}", "Gaussians": int(np.count_nonzero(op < t)),
             "% of total": round(pct(op < t), 2)} for t in OPACITY_THRESHOLDS]
    rows.append({"Opacity threshold": "> 0.9 (nearly opaque)", "Gaussians": int(np.count_nonzero(op > 0.9)),
                 "% of total": round(pct(op > 0.9), 2)})
    return rows


def scale_features(g):
    s = np.sort(g["scale"], axis=1)
    f = {
        "s_max": s[:, 2],
        "s_min": s[:, 0],
        "aniso": s[:, 2] / np.maximum(s[:, 0], 1e-12),
        "flat": s[:, 0] / np.maximum(s[:, 1], 1e-12),
        "thin": s[:, 1] / np.maximum(s[:, 2], 1e-12),
        "volume": s.prod(axis=1),
    }
    f["needle"] = f["thin"] < SHAPE_RATIO
    f["disc"] = (f["flat"] < SHAPE_RATIO) & ~f["needle"]
    f["blob"] = ~f["needle"] & ~f["disc"]
    return f


SHAPE_RATIO = 0.1


def scale_table(g):
    f = scale_features(g)
    q = lambda a, p: float(np.percentile(a, p))
    rows = []
    for key, label in [("s_max", "Longest axis"), ("s_min", "Shortest axis"),
                       ("aniso", "Long/short ratio")]:
        a = f[key]
        rows.append({"Quantity": label, "P1": f"{q(a, 1):.3g}", "P50 (median)": f"{q(a, 50):.3g}",
                     "P99": f"{q(a, 99):.3g}", "Max": f"{a.max():.3g}"})
    extra = [
        ("% needle (mid/long < 0.1)", pct(f["needle"])),
        ("% disc (short/mid < 0.1)", pct(f["disc"])),
        ("% 3D blob (the rest)", pct(f["blob"])),
        ("% near-round (long/short < 2)", pct(f["aniso"] < 2)),
    ]
    for label, v in extra:
        rows.append({"Quantity": label, "P1": "", "P50 (median)": f"{v:.1f} %", "P99": "", "Max": ""})
    return rows


def sh_band_rms(g):
    out = {0: np.sqrt(np.mean(g["f_dc"] ** 2, axis=1))}
    for band, sl in SH_BANDS.items():
        c = g["f_rest"][:, sl, :]
        if c.shape[1]:
            out[band] = np.sqrt(np.mean(c ** 2, axis=(1, 2)))
    return out


SH_EPS = 0.01


def sh_table(g):
    rms = sh_band_rms(g)
    energy = {b: float(np.sum(v ** 2) * (3 if b == 0 else 3 * (2 * b + 1))) for b, v in rms.items()}
    total = sum(energy.values())
    rows = []
    for b, v in rms.items():
        rows.append({"SH degree": b, "Coefficients (3 channels)": 3 * (2 * b + 1),
                     "% color energy": round(100 * energy[b] / total, 2),
                     "Median RMS": f"{np.median(v):.4f}",
                     f"% Gaussians RMS < {SH_EPS}": round(pct(v < SH_EPS), 1)})
    return rows


def sh_zero_pct(g):
    return pct(np.all(g["f_rest"] == 0, axis=(1, 2)))


SH_EPS_LIST = [0.01, 0.02, 0.05, 0.1]


def sh_threshold_table(g):
    rows = []
    for e in SH_EPS_LIST:
        need = sh_needed_degree(g, e)
        row = {"RMS threshold": e}
        for d in range(4):
            row[f"% need only degree {d}"] = round(pct(need == d), 1)
        rows.append(row)
    return rows


def sh_needed_degree(g, eps=SH_EPS):
    rms = sh_band_rms(g)
    need = np.zeros(g["N"], np.int8)
    for b in sorted(k for k in rms if k > 0):
        need[rms[b] >= eps] = b
    return need


def spatial_table(g):
    xyz = g["xyz"]
    lo, hi = np.percentile(xyz, 1, axis=0), np.percentile(xyz, 99, axis=0)
    span = hi - lo
    outside = np.any((xyz < lo - 0.5 * span) | (xyz > hi + 0.5 * span), axis=1)
    rows = []
    for i, ax in enumerate("xyz"):
        rows.append({"Axis": ax, "Min": f"{xyz[:, i].min():.2f}", "P1": f"{lo[i]:.2f}",
                     "P99": f"{hi[i]:.2f}", "Max": f"{xyz[:, i].max():.2f}", "P1-P99 range": f"{span[i]:.2f}"})
    rows.append({"Axis": "Outside main region", "Min": "", "P1": "", "P99": "", "Max": "",
                 "P1-P99 range": f"{pct(outside):.2f} % of Gaussians"})
    return rows


def spatial_concentration(g, fracs=(0.02, 0.05, 0.1)):
    xyz = g["xyz"]
    lo, hi = np.percentile(xyz, 1, axis=0), np.percentile(xyz, 99, axis=0)
    H, e0, e2 = np.histogram2d(xyz[:, 0], xyz[:, 2], bins=300, range=[[lo[0], hi[0]], [lo[2], hi[2]]])
    i, j = np.unravel_index(H.argmax(), H.shape)
    c = np.array([(e0[i] + e0[i + 1]) / 2, np.median(xyz[:, 1]), (e2[j] + e2[j + 1]) / 2])
    diag = np.linalg.norm(hi - lo)
    d = np.linalg.norm(xyz - c, axis=1)
    rows = []
    for f in fracs:
        r = f * diag
        rows.append({"Radius (scene units)": f"{r:.1f}",
                     "% Gaussians inside": round(pct(d < r), 1),
                     "% of P1-P99 box volume": f"{100 * (4 / 3 * np.pi * r ** 3) / np.prod(hi - lo):.2f}"})
    return rows


def quantize_uniform(a, bits):
    lo, hi = a.min(), a.max()
    levels = 2 ** bits - 1
    if hi == lo:
        return a.copy()
    q = np.round((a - lo) / (hi - lo) * levels)
    return (q / levels * (hi - lo) + lo).astype(np.float32)


def quant_table(g):
    targets = {
        "Position (scene units)": (g["xyz"], lambda a: a),
        "Base color f_dc": (g["f_dc"], lambda a: a),
        "Higher-order SH f_rest": (g["f_rest"], lambda a: a),
        "Opacity (after sigmoid)": (g["opacity_raw"], lambda a: 1 / (1 + np.exp(-a))),
        "Scale (after exp)": (g["scale_log"], np.exp),
        "Rotation (quaternion)": (g["rot"], lambda a: a),
    }
    rows = []
    for name, (a, act) in targets.items():
        if a.size == 0:
            continue
        ref = act(a.astype(np.float64))
        scale = np.std(ref) + 1e-12
        row = {"Group": name, "Std dev": f"{scale:.3g}"}
        for label, aq in [("float16", a.astype(np.float16).astype(np.float32)),
                          ("8-bit", quantize_uniform(a, 8))]:
            err = np.abs(act(aq.astype(np.float64)) - ref)
            row[f"{label}: mean error"] = f"{err.mean():.3g}"
            row[f"{label}: error/std dev"] = f"{err.mean() / scale:.2%}"
        rows.append(row)
    return rows


def position_quant_table(g):
    xyz = g["xyz"]
    size = float(np.median(scale_features(g)["s_max"]))
    variants = [
        ("float16", xyz.astype(np.float16).astype(np.float32), 16),
        ("8-bit uniform", np.stack([quantize_uniform(xyz[:, k], 8) for k in range(3)], 1), 8),
        ("16-bit uniform", np.stack([quantize_uniform(xyz[:, k], 16) for k in range(3)], 1), 16),
    ]
    rows = []
    for name, q, bits in variants:
        e = np.linalg.norm(q - xyz, axis=1)
        rows.append({"Position encoding": name, "Bits / coordinate": bits,
                     "Mean error": f"{e.mean():.4f}", "Max error": f"{e.max():.4f}",
                     "Mean error / median Gaussian size": f"{e.mean() / size:.2f}x"})
    return rows, size


def summary_row(label, g):
    f = scale_features(g)
    return {"File": label, "Gaussians": f"{g['N']:,}", "Size (MB)": round(g["file_bytes"] / MB, 1),
            "Mean opacity": round(float(g["opacity"].mean()), 3),
            "% opacity < 0.05": round(pct(g["opacity"] < 0.05), 1),
            "Median longest axis": f"{np.median(f['s_max']):.3g}",
            "% needle": round(pct(f["needle"]), 1),
            "% disc": round(pct(f["disc"]), 1)}


def whatif_table(g):
    n = g["N"]
    op = g["opacity"]
    need = sh_needed_degree(g)
    base = 62 * 4 * n

    def floats_for(deg, normals=False):
        return 3 + (3 if normals else 0) + 3 * (deg + 1) ** 2 + 1 + 3 + 4

    keep005 = np.count_nonzero(op >= 0.005)
    keep05 = np.count_nonzero(op >= 0.05)
    adaptive = sum((floats_for(d) * 4 + 1) * int(np.count_nonzero(need == d)) for d in range(4))

    scen = [
        ("0. Original file", base),
        ("1. Drop normals", floats_for(3) * 4 * n),
        ("2. (1) + drop SH degree 3", floats_for(2) * 4 * n),
        ("3. (1) + keep only SH degrees 0-1", floats_for(1) * 4 * n),
        ("4. (1) + keep only base color (degree 0)", floats_for(0) * 4 * n),
        ("5. (1) + float16 everywhere", floats_for(3) * 2 * n),
        ("6. Prune opacity < 0.005", 62 * 4 * keep005),
        ("7. Prune opacity < 0.05", 62 * 4 * keep05),
        (f"8. (1) + adaptive SH (threshold {SH_EPS})", adaptive),
        ("9. Combined: prune<0.05 + drop normals + SH degree 2 + float16", floats_for(2) * 2 * keep05),
    ]
    return [{"Scenario": s, "Size (MB)": round(b / MB, 1), "Compression (x)": round(base / b, 1),
             "% remaining": round(100 * b / base, 1)} for s, b in scen]
