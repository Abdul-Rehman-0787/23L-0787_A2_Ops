import os, hashlib
from pathlib import Path
import pandas as pd
from PIL import Image

ROOT = Path("data/raw")
rows = []

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

for cls_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    for f in cls_dir.rglob("*"):
        if not f.is_file():
            continue
        row = {"path": str(f), "class": cls_dir.name,
               "file_size": f.stat().st_size, "sha256": sha256(f),
               "width": None, "height": None, "mode": None,
               "format": None, "corrupt": False}
        try:
            with Image.open(f) as im:
                im.verify()                 # checks file integrity
            with Image.open(f) as im:       # re-open: verify() invalidates it
                row["width"], row["height"] = im.size
                row["mode"], row["format"] = im.mode, im.format
        except Exception:
            row["corrupt"] = True
        rows.append(row)

df = pd.DataFrame(rows)
df["duplicate"] = df.duplicated("sha256", keep="first")
df.to_csv("data/audit.csv", index=False)

print("Counts per class:\n", df["class"].value_counts(), "\n")
print("Corrupt files:", df["corrupt"].sum())
print("Exact duplicates:", df["duplicate"].sum())
print(df.groupby("class")[["width", "height", "file_size"]]
        .agg(["min", "median", "max"]).T)
print("\nModes:\n", df["mode"].value_counts())
print("\nFormats:\n", df["format"].value_counts())