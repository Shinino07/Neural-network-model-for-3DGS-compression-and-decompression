import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from .stats import save_csv

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "figure.dpi": 110, "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
})

FINDINGS = os.path.join(config.OUT_DIR, "findings.json")


def save_fig(fig, name):
    path = os.path.join(config.FIG_DIR, name + ".png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("  figure ->", path)


def save_table(rows, name):
    path = os.path.join(config.TAB_DIR, name + ".csv")
    save_csv(rows, path)
    print("  table  ->", path)


def add_findings(section, lines):
    data = {}
    if os.path.exists(FINDINGS):
        with open(FINDINGS, encoding="utf-8") as f:
            data = json.load(f)
    data[section] = lines
    with open(FINDINGS, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    for l in lines:
        print("  *", l)
