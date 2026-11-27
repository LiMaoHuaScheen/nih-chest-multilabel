# -*- coding: utf-8 -*-
"""
gradcam.py - Grad-CAM heatmap visualization for NIH chest multi-label model.
Uses best checkpoint (baseline arm). Picks a few test images covering different diseases,
computes Grad-CAM for the predicted top-1 disease + a chosen reference class, overlays heatmap.
"""
import os, csv, argparse, random
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

CLS = ["Atelectasis","Cardiomegaly","Consolidation","Edema","Effusion","Emphysema",
       "Fibrosis","Hernia","Infiltration","Mass","Nodule","Pleural_Thickening",
       "Pneumonia","Pneumothorax"]
NCLS=len(CLS)
DATA=r"data"
IMGDIR=os.path.join(DATA,"images")
OUT=r"output_chest"
CKPT=os.path.join(OUT,"baseline_best.pt")

def build():
    m = models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, NCLS)
    m.load_state_dict(torch.load(CKPT, map_location="cpu"))
    m.eval()
    return m

tf = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def gradcam(model, x_in, class_idx):
    x = x_in.clone().detach().requires_grad_(True)
    feats=[]; grads=[]
    def fwd(mod,inp,out): feats.append(out)
    def bwd(mod,gin,gout): grads.append(gout[0])
    # hook on denseblock4 (last feature block before norm+relu+pool)
    h1=model.features.denseblock4.register_forward_hook(fwd)
    h2=model.features.denseblock4.register_full_backward_hook(bwd)
    model.zero_grad()
    out=model(x)
    score=out[0,class_idx]
    model.zero_grad()
    score.backward()
    A=feats[0].detach()[0]       # (C,7,7)
    G=grads[0].detach()[0]       # (C,7,7)
    w=G.mean(dim=(1,2))          # (C,)
    cam=(w[:,None,None]*A).sum(0)
    cam=F.relu(cam)
    cam=(cam-cam.min())/(cam.max()-cam.min()+1e-8)
    cam=torch.nn.functional.interpolate(cam[None,None], size=(224,224), mode="bilinear", align_corners=False)[0,0]
    cam=cam.detach().cpu().numpy()
    h1.remove(); h2.remove()
    return cam

def main(n_samples=6, seed=42):
    random.seed(seed); np.random.seed(seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=build().to(device)
    # load test.csv
    rows=[]
    with open(os.path.join(DATA,"test.csv"), encoding="utf-8") as f:
        rd=csv.DictReader(f)
        for r in rd:
            labs=[l for l in r["labels"].split("|") if l]
            rows.append((r["image"],labs))
    # pick a spread of samples: at least 1 each from 4 rare + 2 common diseases
    rare_pick = ["Hernia","Pneumothorax","Nodule","Edema"]
    common_pick = ["Atelectasis","Effusion"]
    picks=[]
    for tgt in rare_pick+common_pick:
        cand=[r for r in rows if tgt in r[1]]
        if cand: picks.append(cand[random.randrange(len(cand))])
    # dedupe by image name (limit n_samples)
    seen=set(); final=[]
    for img,labs in picks:
        if img not in seen:
            seen.add(img); final.append((img,labs))
            if len(final)>=n_samples: break

    fig, axes = plt.subplots(2, len(final), figsize=(3*len(final), 6))
    if len(final)==1: axes=axes.reshape(2,1)
    for j,(img,labs) in enumerate(final):
        raw=Image.open(os.path.join(IMGDIR,img)).convert("RGB").resize((224,224))
        x=tf(raw)[None].to(device); x.requires_grad_(True)
        with torch.no_grad():
            p=torch.sigmoid(model(x))[0].cpu().numpy()
        top=int(np.argmax(p))
        cam=gradcam(model, x.clone().detach().requires_grad_(True), top)
        heat=cm.jet(cam)[...,:3]
        overlay=0.55*np.array(raw)/255.0 + 0.45*heat
        axes[0,j].imshow(raw); axes[0,j].set_title("Original\n"+",".join(labs[:3]), fontsize=9); axes[0,j].axis("off")
        axes[1,j].imshow(overlay); axes[1,j].set_title(f"Grad-CAM\npred={CLS[top]}\np={p[top]:.2f}", fontsize=9); axes[1,j].axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(OUT,"gradcam.png"), dpi=120, bbox_inches="tight"); plt.close()
    print("GRADCAM DONE ->", os.path.join(OUT,"gradcam.png"))
    print("Samples:", [(i,l) for i,l in final])

if __name__=="__main__":
    main()
