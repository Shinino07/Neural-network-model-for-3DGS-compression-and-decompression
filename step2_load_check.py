import os
import numpy as np

import config
from gslib.load import load_gaussians, read_header
from gslib.out import save_table, add_findings

props, n, header_bytes, fmt = read_header(config.PLY_MAIN)
raw, g = load_gaussians(config.PLY_MAIN)
expected = header_bytes + n * raw.dtype.itemsize

print(f"File: {config.PLY_MAIN}")
print(f"  N = {g['N']:,} Gaussians, {len(props)} properties, SH degree = {g['sh_degree']}")

checks = [
    ("File size = header + N x bytes/Gaussian", g["file_bytes"] == expected),
    ("No NaN / infinity", all(np.isfinite(raw[k]).all() for k in raw.dtype.names)),
    ("Normals are all 0 (unused)", all(np.all(raw[k] == 0) for k in ("nx", "ny", "nz") if k in raw.dtype.names)),
    ("Opacity after sigmoid lies in (0, 1)", bool(((g["opacity"] >= 0) & (g["opacity"] <= 1)).all())),
    ("No zero-length quaternions", bool((g["rot_norm_raw"] > 1e-8).all())),
]
for name, ok in checks:
    print(f"  [{'OK  ' if ok else 'FAIL'}] {name}")

rows = [
    {"Item": "Scene", "Value": config.SCENE},
    {"Item": "File", "Value": os.path.basename(config.PLY_MAIN)},
    {"Item": "Number of Gaussians N", "Value": f"{g['N']:,}"},
    {"Item": "Properties / Gaussian", "Value": len(props)},
    {"Item": "Bytes / Gaussian", "Value": raw.dtype.itemsize},
    {"Item": "SH degree", "Value": g["sh_degree"]},
    {"Item": "File size", "Value": f"{g['file_bytes'] / 1024**2:.1f} MB"},
] + [{"Item": "Check: " + k, "Value": "OK" if ok else "FAIL"} for k, ok in checks]
save_table(rows, "overview")

add_findings("overview", [
    f"{g['N']:,} Gaussians, each packing {len(props)} float32 values ({raw.dtype.itemsize} bytes).",
    f"{g['file_bytes'] / 1024**2:.1f} MB in total, and "
    + ("the math checks out to the byte: header + N x bytes per Gaussian." if checks[0][1]
       else "the math does NOT match header + N x bytes per Gaussian. Something is off and needs a closer look."),
])
