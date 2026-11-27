# -*- coding: utf-8 -*-
"""
plot_chest.py - Read output_chest/all_results.json, emit:
  - curve_auc.png:   val_auc per-epoch curves for 3 arms
  - bar_arms.png:    test_auc + test_f1 grouped bar for 3 arms
  - bar_per_class.png: per-class AUC for best arm (baseline)
Output to output_chest/. English labels only (matplotlib default font has no CJK).
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = r"output_chest"
CLS = ["Atelectasis","Cardiomegaly","Consolidation","Edema","Effusion","Emphysema",
       "Fibrosis","Hernia","Infiltration","Mass","Nodule","Pleural_Thickening",
       "Pneumonia","Pneumothorax"]

with open(os.path.join(OUT,"all_results.json")) as f:
    res = json.load(f)
arms = [r["arm"] for r in res]
colors = {"baseline":"#4C72B0","focal":"#DD8452","weighted":"#55A868"}

# ---- 1. val_auc curves ----
plt.figure(figsize=(9,5.5))
for r in res:
    plt.plot(range(1,len(r["history"]["val_auc"])+1), r["history"]["val_auc"],
             marker="o", label=r["arm"], color=colors[r["arm"]])
plt.xlabel("Epoch"); plt.ylabel("Validation macro-AUC")
plt.title("NIH Chest X-ray - Validation AUC per Epoch")
plt.grid(True, alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT,"curve_auc.png"), dpi=120); plt.close()

# ---- 2. test_auc + test_f1 grouped bar ----
x = np.arange(len(arms)); w=0.35
fig, ax = plt.subplots(figsize=(8,5))
aucs=[r["test_auc"] for r in res]; f1s=[r["test_f1"] for r in res]
b1=ax.bar(x-w/2, aucs, w, label="Test macro-AUC", color="#4C72B0")
b2=ax.bar(x+w/2, f1s, w, label="Test macro-F1 (thr=0.5)", color="#DD8452")
ax.set_xticks(x); ax.set_xticklabels(arms)
ax.set_ylabel("Score"); ax.set_title("NIH Chest - Test metrics by loss arm")
ax.set_ylim(0,1)
for bars in [b1,b2]:
    for b in bars:
        ax.annotate(f"{b.get_height():.3f}", (b.get_x()+b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontsize=9)
ax.legend(); ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"bar_arms.png"), dpi=120); plt.close()

# ---- 3. per-class AUC for each arm ----
fig, ax = plt.subplots(figsize=(12,5.5))
x = np.arange(len(CLS)); w = 0.27
for i,r in enumerate(res):
    ax.bar(x+(i-1)*w, r["per_class_auc"], w, label=r["arm"], color=colors[r["arm"]])
ax.set_xticks(x); ax.set_xticklabels(CLS, rotation=45, ha="right")
ax.set_ylabel("Test AUC"); ax.set_ylim(0,1)
ax.set_title("NIH Chest - Per-class AUC (higher = better disease detection)")
ax.axhline(0.5, color="grey", linestyle="--", alpha=0.5)
ax.legend(); ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"bar_per_class.png"), dpi=120); plt.close()

print("PLOTS DONE ->", OUT);
for f in ["curve_auc.png","bar_arms.png","bar_per_class.png"]:
    p=os.path.join(OUT,f); print(f, os.path.getsize(p),"bytes")
