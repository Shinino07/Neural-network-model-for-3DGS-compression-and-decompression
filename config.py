import os

SCENE = "truck"

PLY_MAIN = os.environ.get("GS_PLY_MAIN", "data/truck_30000.ply")
PLY_EARLY = os.environ.get("GS_PLY_EARLY", "data/truck_7000.ply")

OUT_DIR = os.environ.get("GS_OUT", "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figs")
TAB_DIR = os.path.join(OUT_DIR, "tables")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)
