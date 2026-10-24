import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

SEED = 42
df = pd.read_csv("data/audit.csv")
print("Total rows:", len(df))

df = df[~df["corrupt"]]
# same hash appearing under BOTH classes = contradictory labels, drop all copies
n_cls = df.groupby("sha256")["class"].nunique()
df = df[~df["sha256"].isin(n_cls[n_cls > 1].index)]
df = df.drop_duplicates("sha256", keep="first")

df["path"] = df["path"].str.replace("\\", "/", regex=False)
df["label"] = df["class"].str.lower().str.startswith("ai").astype(int)
print(df.groupby(["class", "label"]).size())
assert df["label"].nunique() == 2, "check class folder names"

trainval, test = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=SEED)
train, val = train_test_split(trainval, test_size=0.15 / 0.85, stratify=trainval["label"], random_state=SEED)

Path("data/splits").mkdir(parents=True, exist_ok=True)
cols = ["path", "label", "class", "width", "height", "file_size", "sha256"]
for name, part in [("train", train), ("val", val), ("test", test)]:
    part[cols].to_csv(f"data/splits/{name}.csv", index=False)
    print(name, len(part), "| AI share:", round(part["label"].mean(), 3))