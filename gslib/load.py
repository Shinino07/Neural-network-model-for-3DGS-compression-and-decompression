import os
import numpy as np

SH_C0 = 0.28209479177387814

SH_BANDS = {1: slice(0, 3), 2: slice(3, 8), 3: slice(8, 15)}

PLY_TYPES = {"float": "f4", "float32": "f4", "double": "f8", "uchar": "u1",
             "uint8": "u1", "int": "i4", "int32": "i4", "uint": "u4"}


def read_header(path):
    props, n, fmt = [], 0, None
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("element") and n:
                raise ValueError("File has elements other than vertex - not supported")
            elif line.startswith("property"):
                _, typ, name = line.split()
                props.append((name, PLY_TYPES[typ]))
            elif line == "end_header":
                break
        header_bytes = f.tell()
    return props, n, header_bytes, fmt


def load_raw(path):
    props, n, header_bytes, fmt = read_header(path)
    if fmt != "binary_little_endian":
        raise ValueError(f"Only binary_little_endian is supported, file is {fmt}")
    dtype = np.dtype([(name, "<" + t) for name, t in props])
    raw = np.fromfile(path, dtype=dtype, count=n, offset=header_bytes)
    return raw


def _stack(raw, names):
    return np.stack([raw[k].astype(np.float32) for k in names], axis=1)


def decode(raw):
    names = raw.dtype.names
    g = {"N": len(raw)}
    g["xyz"] = _stack(raw, ["x", "y", "z"])
    g["f_dc"] = _stack(raw, ["f_dc_0", "f_dc_1", "f_dc_2"])
    g["rgb"] = np.clip(0.5 + SH_C0 * g["f_dc"], 0, 1)

    n_rest = sum(1 for k in names if k.startswith("f_rest_"))
    if n_rest:
        rest = _stack(raw, [f"f_rest_{i}" for i in range(n_rest)])
        g["f_rest"] = rest.reshape(-1, 3, n_rest // 3).transpose(0, 2, 1)
    else:
        g["f_rest"] = np.zeros((g["N"], 0, 3), np.float32)
    g["sh_degree"] = int(round(np.sqrt(n_rest // 3 + 1))) - 1

    g["opacity_raw"] = raw["opacity"].astype(np.float32)
    g["opacity"] = 1.0 / (1.0 + np.exp(-g["opacity_raw"]))

    g["scale_log"] = _stack(raw, ["scale_0", "scale_1", "scale_2"])
    g["scale"] = np.exp(g["scale_log"])

    q = _stack(raw, ["rot_0", "rot_1", "rot_2", "rot_3"])
    g["rot_norm_raw"] = np.linalg.norm(q, axis=1)
    g["rot"] = q / np.maximum(g["rot_norm_raw"][:, None], 1e-12)
    return g


def load_gaussians(path):
    raw = load_raw(path)
    g = decode(raw)
    g["path"] = path
    g["file_bytes"] = os.path.getsize(path)
    return raw, g


def property_groups(names):
    groups = {
        "Position (xyz)": ["x", "y", "z"],
        "Normal (unused)": ["nx", "ny", "nz"],
        "Base color (SH degree 0)": [k for k in names if k.startswith("f_dc_")],
    }
    rest = [k for k in names if k.startswith("f_rest_")]
    per_ch = len(rest) // 3
    for band, sl in SH_BANDS.items():
        idx = [c * per_ch + i for c in range(3) for i in range(per_ch)[sl]]
        if idx:
            groups[f"SH degree {band}"] = [f"f_rest_{i}" for i in idx]
    groups["Opacity"] = ["opacity"]
    groups["Scale"] = [k for k in names if k.startswith("scale_")]
    groups["Rotation"] = [k for k in names if k.startswith("rot_")]
    return {k: [n for n in v if n in names] for k, v in groups.items() if v}


def save_ply(path, raw):
    rev = {v: k for k, v in PLY_TYPES.items() if k in ("float", "double", "uchar", "int", "uint")}
    lines = ["ply", "format binary_little_endian 1.0", f"element vertex {len(raw)}"]
    for name in raw.dtype.names:
        lines.append(f"property {rev[raw.dtype[name].str[1:]]} {name}")
    lines.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(lines) + "\n").encode("ascii"))
        raw.astype(raw.dtype.newbyteorder("<")).tofile(f)
