# -*- coding: utf-8 -*-
"""
train_chest.py - NIH ChestX-ray14 multi-label classification.
3 arms: baseline(BCE) / focal / weighted-sampler.
Backbone: DenseNet-121 (ImageNet pretrained), imgsz 224, Adam 1e-4.
Speedups: AMP bf16 autocast + GradScaler, batch 64, num_workers 4.
Eval: macro-AUC (primary), macro-F1, per-class AUC.
"""
import os, csv, json, argparse, random, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score

CLS = ["Atelectasis","Cardiomegaly","Consolidation","Edema","Effusion","Emphysema",
       "Fibrosis","Hernia","Infiltration","Mass","Nodule","Pleural_Thickening",
       "Pneumonia","Pneumothorax"]
NCLS = len(CLS)
DATA = r"data"
IMGDIR = os.path.join(DATA, "images")

class ChestDS(Dataset):
    def __init__(self, csvf, train=False):
        self.items = []
        with open(csvf, encoding="utf-8") as f:
            rd = csv.DictReader(f)
            for r in rd:
                mh = [int(x) for x in r["multi_hot"].split(",")]
                self.items.append((r["image"], np.array(mh, dtype=np.float32)))
        self.train = train
        self.tf_tr = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.1,0.1,0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        self.tf_va = transforms.Compose([
            transforms.Resize((224,224)), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        img, mh = self.items[i]
        x = Image.open(os.path.join(IMGDIR, img)).convert("RGB")
        x = (self.tf_tr if self.train else self.tf_va)(x)
        return x, torch.tensor(mh)

def build_model():
    m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    m.classifier = nn.Linear(m.classifier.in_features, NCLS)
    return m

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-bce)
    return (alpha*(1-pt)**gamma*bce).mean()

def multilabel_sampler(dataset):
    freq = np.zeros(NCLS)
    for _, mh in dataset.items: freq += mh
    freq = np.maximum(freq, 1)
    w = []
    for _, mh in dataset.items:
        pos = mh.astype(bool)
        w.append(1.0/np.min(freq[pos]) if pos.any() else 0.0)
    return WeightedRandomSampler(w, num_samples=len(w), replacement=True)

def evaluate(model, loader, device):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type=="cuda")):
        for x,y in loader:
            x=x.to(device); p=torch.sigmoid(model(x).float()).cpu().numpy()
            ps.append(p); ys.append(y.numpy())
    ys=np.vstack(ys); ps=np.vstack(ps)
    aucs=[]
    for j in range(NCLS):
        if ys[:,j].sum()>0 and ys[:,j].sum()<len(ys):
            aucs.append(roc_auc_score(ys[:,j], ps[:,j]))
    macro_auc = float(np.mean(aucs)) if aucs else 0.0
    pred = (ps>=0.5).astype(int)
    mf1 = float(f1_score(ys, pred, average="macro", zero_division=0))
    return macro_auc, mf1, aucs

def run_arm(arm, epochs, batch, device, outdir):
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    tr = ChestDS(os.path.join(DATA,"train.csv"), train=True)
    va = ChestDS(os.path.join(DATA,"val.csv"))
    te = ChestDS(os.path.join(DATA,"test.csv"))
    if arm=="weighted":
        sampler = multilabel_sampler(tr)
        tr_loader = DataLoader(tr, batch_size=batch, sampler=sampler, num_workers=4, pin_memory=True, persistent_workers=True)
        criterion = nn.BCEWithLogitsLoss()
    else:
        tr_loader = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        criterion = nn.BCEWithLogitsLoss() if arm=="baseline" else "focal"
    va_loader = DataLoader(va, batch_size=batch, num_workers=2, pin_memory=True)
    te_loader = DataLoader(te, batch_size=batch, num_workers=2, pin_memory=True)

    model = build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type=="cuda"))
    hist={"train_loss":[],"val_auc":[],"val_f1":[],"epoch_time_s":[]}
    best_auc=0.0; best_ep=0; patience=3; bad=0
    for ep in range(1, epochs+1):
        t0=time.time()
        model.train(); tot=0.0; nb=0
        for x,y in tr_loader:
            x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type=="cuda")):
                logits=model(x)
                loss = criterion(logits,y) if criterion!="focal" else focal_loss(logits,y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot+=loss.item(); nb+=1
        sched.step()
        dt=time.time()-t0
        auc,f1,_ = evaluate(model, va_loader, device)
        hist["train_loss"].append(tot/nb)
        hist["val_auc"].append(auc); hist["val_f1"].append(f1)
        hist["epoch_time_s"].append(round(dt,1))
        print(f"[{arm}] ep{ep} loss={tot/nb:.4f} val_auc={auc:.4f} val_f1={f1:.4f} time={dt:.1f}s", flush=True)
        if auc>best_auc:
            best_auc=auc; best_ep=ep; bad=0
            torch.save(model.state_dict(), os.path.join(outdir,f"{arm}_best.pt"))
        else:
            bad+=1
            if bad>=patience:
                print(f"[{arm}] early stop at ep{ep}"); break
    model.load_state_dict(torch.load(os.path.join(outdir,f"{arm}_best.pt"), map_location=device))
    tauc, tf1, percls = evaluate(model, te_loader, device)
    res = {"arm":arm,"best_epoch":best_ep,"best_val_auc":best_auc,
           "test_auc":tauc,"test_f1":tf1,"per_class_auc":[round(a,4) for a in percls],
           "history":hist}
    with open(os.path.join(outdir,f"{arm}_result.json"),"w") as f:
        json.dump(res, f, indent=2)
    print(f"[{arm}] TEST auc={tauc:.4f} f1={tf1:.4f} (best_ep={best_ep})", flush=True)
    return res

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=8)
    ap.add_argument("--batch",type=int,default=64)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--arms",default="baseline,focal,weighted")
    args=ap.parse_args()
    device=torch.device(args.device if torch.cuda.is_available() else "cpu")
    outdir=r"output_chest"
    os.makedirs(outdir, exist_ok=True)
    all_res=[]
    for arm in args.arms.split(","):
        all_res.append(run_arm(arm, args.epochs, args.batch, device, outdir))
    with open(os.path.join(outdir,"all_results.json"),"w") as f:
        json.dump(all_res, f, indent=2)
    print("ALL ARMS DONE", flush=True)

if __name__=="__main__":
    main()
