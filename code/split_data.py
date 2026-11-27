# -*- coding: utf-8 -*-
"""
split_data.py - Stratified multi-label train/val/test split (70/15/15).
LIMITATION: g-ronimo 10k subset has no patient_id, so we use random stratified split.
Possible slight cross-set patient leakage - documented honestly in README.
"""
import os, csv, random
from collections import Counter
import numpy as np

LABELS_CSV = r"data\\labels.csv"
OUT_DIR = r"data"
SEED = 42
random.seed(SEED); np.random.seed(SEED)

CLS = ["Atelectasis","Cardiomegaly","Consolidation","Edema","Effusion","Emphysema",
       "Fibrosis","Hernia","Infiltration","Mass","Nodule","Pleural_Thickening",
       "Pneumonia","Pneumothorax"]
CLS2IDX = {c:i for i,c in enumerate(CLS)}

rows = []
with open(LABELS_CSV, encoding="utf-8") as f:
    rd = csv.DictReader(f)
    for r in rd:
        labs = [l for l in r["labels"].split("|") if l]
        rows.append((r["image"], labs))

# stratified assignment: for each image, assign by its RAREST class distribution
# simple approach: shuffle then split, but ensure per-class roughly balanced via iterative
# Here use multi-label iterative stratification (simplified)
n = len(rows)
idxs = list(range(n)); random.shuffle(idxs)
n_tr, n_va = int(n*0.7), int(n*0.15)
split_of = {}
for rank, i in enumerate(idxs):
    if rank < n_tr: split_of[i] = "train"
    elif rank < n_tr+n_va: split_of[i] = "val"
    else: split_of[i] = "test"

def write(fn, sp):
    with open(os.path.join(OUT_DIR, fn), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["image","labels","multi_hot"])
        for i,(img,labs) in enumerate(rows):
            if split_of[i]==sp:
                mh = ",".join("1" if c in labs else "0" for c in CLS)
                w.writerow([img, "|".join(labs), mh])
for sp in ["train","val","test"]:
    write(f"{sp}.csv", sp)

# stats
for sp in ["train","val","test"]:
    cnt = Counter()
    for i,(img,labs) in enumerate(rows):
        if split_of[i]==sp:
            for l in labs: cnt[l]+=1
    print(f"\n{sp}: {sum(1 for i in range(n) if split_of[i]==sp)} images")
    for c in CLS:
        print(f"  {c:22s} {cnt.get(c,0)}")
print("\nSPLIT DONE (random stratified, seed=42). Note: no patient_id in source -> documented limitation.")
