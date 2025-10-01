# -*- coding: utf-8 -*-
# Train on sdoh_train.csv with: RoBERTa + mean pooling over groups-of-5 + MLP(768→128→16→4, BN+GeLU)
# Group-level split 70/10/20, class-weighted CE, per-epoch loss & weighted F1.

import os, math, random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from transformers import AutoTokenizer, AutoModel, get_scheduler  
import torch.optim as optim
import matplotlib.pyplot as plt

# -------------------- Config & Utils --------------------
@dataclass
class TrainConfig:
    save_path: str = "roberta_meanpool_best.pt"
    roberta_name: str = "roberta-base"
    max_length: int = 256
    batch_size: int = 8            # batch = groups; each group has 5 posts internally
    epochs: int = 20
    lr_head: float = 1e-3
    lr_backbone: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    unfreeze_last_n_layers: int = 0   # 0=freeze all; e.g., 3=unfreeze last 3 layers
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2
    pin_memory: bool = True
    data_csv: str = "sdoh_train.csv"  # must be in same dir

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# -------------------- Data Loading --------------------
def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "post_sequence" not in df.columns or "suicide_risk" not in df.columns:
        raise KeyError("需要列 post_sequence(文本) 和 suicide_risk(标签)。")
    # 只保留需要的两列，并重命名为 text / suicide_risk
    df = df[["post_sequence", "suicide_risk"]].rename(columns={"post_sequence": "text"})
    # 基本校验
    if len(df) % 5 != 0:
        raise ValueError(f"行数 {len(df)} 不是 5 的倍数。确保原始顺序未被打乱。")
    # 确保每5行同一标签
    G = len(df) // 5
    for g in range(G):
        labs = df.iloc[g*5:(g+1)*5]["suicide_risk"].tolist()
        if len(set(labs)) != 1:
            raise ValueError(f"第 {g} 组 5 条的标签不一致：{labs}")
    # 强制标签为 int
    df["suicide_risk"] = df["suicide_risk"].astype(int)
    return df

# -------------------- Dataset & Collator --------------------
class GroupedTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, group_indices: List[int]):
        self.df = df.reset_index(drop=True)
        self.group_indices = list(group_indices)

    def __len__(self): return len(self.group_indices)

    def __getitem__(self, idx):
        g = self.group_indices[idx]
        s, e = g*5, g*5+5
        rows = self.df.iloc[s:e]
        texts = rows["text"].tolist()
        labels = rows["suicide_risk"].tolist()
        if len(set(labels)) != 1:
            raise ValueError(f"Group {g} label mismatch: {labels}")
        return {"texts": texts, "label": int(labels[0])}

class Collator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 256):
        self.tokenizer = tokenizer; self.max_length = max_length
    def __call__(self, batch):
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        flat_texts, sizes = [], []
        for b in batch:
            flat_texts.extend(b["texts"]); sizes.append(len(b["texts"]))  # should be 5
        enc = self.tokenizer(
            flat_texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        return enc, labels, torch.tensor(sizes, dtype=torch.long)

# -------------------- Model --------------------
class RobertaMeanPoolClassifier(nn.Module):
    def __init__(self, roberta_name="roberta-base", num_classes=4):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(roberta_name)
        hidden = self.backbone.config.hidden_size  # 768 for roberta-base
        self.head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 16),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Linear(16, num_classes),
        )
    @torch.no_grad()
    def _mean_pool_groups(self, cls_batch: torch.Tensor, group_sizes: torch.Tensor):
        # cls_batch: (B*5, 768) from CLS token; group_sizes: (B,) each==5
        out, off = [], 0
        for k in group_sizes.tolist():
            out.append(cls_batch[off:off+k].mean(dim=0, keepdim=True)); off += k
        return torch.cat(out, dim=0)  # (B, 768)
    def forward(self, encodings, group_sizes: torch.Tensor):
        out = self.backbone(input_ids=encodings["input_ids"], attention_mask=encodings["attention_mask"])
        cls = out.last_hidden_state[:, 0, :]  # (B*5, 768)
        pooled = self._mean_pool_groups(cls, group_sizes)  # (B, 768)
        return self.head(pooled)  # (B, C)

def set_backbone_trainable(model: RobertaMeanPoolClassifier, unfreeze_last_n_layers: int = 0):
    for p in model.backbone.parameters(): p.requires_grad = False
    if unfreeze_last_n_layers > 0:
        encoder_layers = model.backbone.encoder.layer  # RobertaEncoder
        for layer in encoder_layers[-unfreeze_last_n_layers:]:
            for p in layer.parameters(): p.requires_grad = True

# -------------------- Train / Eval --------------------
def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    classes = np.arange(num_classes)
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return torch.tensor(w, dtype=torch.float32)

@torch.no_grad()
def evaluate(model, loader, device, criterion=None) -> Dict[str, float]:
    model.eval()
    all_logits, all_labels, total_loss, n = [], [], 0.0, 0
    for enc, labels, sizes in loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        labels = labels.to(device); sizes = sizes.to(device)
        logits = model(enc, sizes)
        all_logits.append(logits.detach().cpu()); all_labels.append(labels.detach().cpu())
        if criterion is not None:
            total_loss += criterion(logits, labels).item() * labels.size(0); n += labels.size(0)
    logits = torch.cat(all_logits); y_true = torch.cat(all_labels).numpy()
    y_pred = logits.argmax(dim=1).numpy()
    wf1 = f1_score(y_true, y_pred, average="weighted")
    out = {"weighted_f1": float(wf1)}
    if criterion is not None and n > 0: out["loss"] = float(total_loss / n)
    return out

def train(df: pd.DataFrame, cfg: TrainConfig, num_classes: int = 4) -> Dict[str, any]:
    set_seed(cfg.seed); device = cfg.device
    # group indices and labels
    G = len(df)//5; group_ids = np.arange(G)
    group_labels = np.array([int(df.iloc[g*5]["suicide_risk"]) for g in group_ids], dtype=int)
    # 70/10/20 split with stratify
    g_tr, g_tmp, y_tr, y_tmp = train_test_split(group_ids, group_labels, test_size=0.3,
                                                random_state=cfg.seed, stratify=group_labels)
    g_val, g_te, y_val, y_te = train_test_split(g_tmp, y_tmp, test_size=2/3,
                                                random_state=cfg.seed, stratify=y_tmp)
    # tokenizer / datasets / loaders
    tok = AutoTokenizer.from_pretrained(cfg.roberta_name)
    collate = Collator(tok, max_length=cfg.max_length)
    ds_tr  = GroupedTextDataset(df, g_tr); ds_val = GroupedTextDataset(df, g_val); ds_te = GroupedTextDataset(df, g_te)
    dl_tr  = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,  num_workers=cfg.num_workers,
                        pin_memory=cfg.pin_memory, collate_fn=collate)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                        pin_memory=cfg.pin_memory, collate_fn=collate)
    dl_te  = DataLoader(ds_te, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                        pin_memory=cfg.pin_memory, collate_fn=collate)
    # model / freeze control
    model = RobertaMeanPoolClassifier(cfg.roberta_name, num_classes=num_classes).to(device)
    set_backbone_trainable(model, cfg.unfreeze_last_n_layers)
    # optimizer with separate LRs
    params = []
    if any(p.requires_grad for p in model.backbone.parameters()):
        params.append({"params": [p for p in model.backbone.parameters() if p.requires_grad],
                       "lr": cfg.lr_backbone, "weight_decay": cfg.weight_decay})
    params.append({"params": list(model.head.parameters()), "lr": cfg.lr_head, "weight_decay": cfg.weight_decay})
    opt = optim.AdamW(params)
    # scheduler (linear warmup → linear decay)
    steps_per_epoch = max(1, math.ceil(len(ds_tr) / cfg.batch_size))
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    sch = get_scheduler(
    "linear",
    optimizer=opt,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
    )
    # class weights from training groups
    class_w = compute_class_weights(y_tr, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    # training loop with early stopping on val weighted F1
    best_f1, best_state, patience, bad = -1.0, None, 5, 0
    hist = {"epoch": [], "train_loss": [], "train_wf1": [], "val_loss": [], "val_wf1": []}
    for ep in range(1, cfg.epochs+1):
        model.train()
        for enc, labels, sizes in dl_tr:
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device); sizes = sizes.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(enc, sizes)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
        # epoch metrics
        tr_eval = evaluate(model, dl_tr, device, criterion)
        val_eval = evaluate(model, dl_val, device, criterion)
        hist["epoch"].append(ep)
        hist["train_loss"].append(tr_eval["loss"]); hist["train_wf1"].append(tr_eval["weighted_f1"])
        hist["val_loss"].append(val_eval["loss"]);   hist["val_wf1"].append(val_eval["weighted_f1"])
        print(f"Epoch {ep:02d} | Train Loss {tr_eval['loss']:.4f} WF1 {tr_eval['weighted_f1']:.4f} | "
              f"Val Loss {val_eval['loss']:.4f} WF1 {val_eval['weighted_f1']:.4f}")
        
        if val_eval["weighted_f1"] > best_f1 + 1e-6:
            best_f1 = val_eval["weighted_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, cfg.save_path)   # 使用 cfg.save_path
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {ep} (best Val WF1={best_f1:.4f}).")
                break
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
        torch.save(best_state, cfg.save_path)

    # final test
    test_eval = evaluate(model, dl_te, device, criterion=None)
    print(f"Test Weighted F1: {test_eval['weighted_f1']:.4f}  (n_groups={len(ds_te)})")
    # plots
    try:
        xs = hist["epoch"]
        plt.figure(); plt.plot(xs, hist["train_loss"], label="Train Loss"); plt.plot(xs, hist["val_loss"], label="Val Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss per Epoch"); plt.legend(); plt.tight_layout()
        plt.savefig("loss_curve.png", dpi=200)
        plt.figure(); plt.plot(xs, hist["train_wf1"], label="Train Weighted F1"); plt.plot(xs, hist["val_wf1"], label="Val Weighted F1")
        plt.xlabel("Epoch"); plt.ylabel("Weighted F1"); plt.title("Weighted F1 per Epoch"); plt.legend(); plt.tight_layout()
        plt.savefig("wf1_curve.png", dpi=200)
    except Exception as e:
        print("Plotting skipped:", e)
    return {"history": hist, "test_eval": test_eval, "class_weights": class_w.detach().cpu().numpy()}

# -------------------- Entry --------------------
if __name__ == "__main__":
    cfg = TrainConfig(
        save_path= "biomed.pt",
        roberta_name="/prj0129/jzh4027/IEEE/local_models/models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract/snapshots/d673b8835373c6fa116d6d8006b33d48734e305d",
        max_length=512,
        batch_size=16,
        epochs=20,
        lr_head=1e-3,
        lr_backbone=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        unfreeze_last_n_layers=4,
        seed=114514,
        data_csv="sdoh_train.csv",
    )
    df = load_dataframe(cfg.data_csv)
    _ = train(df, cfg, num_classes=4)
