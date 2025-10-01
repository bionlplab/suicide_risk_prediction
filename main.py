#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stacking on probabilities
- Train one LSTM per backbone (BioBERT-base / RoBERTa-large / BioMedBERT)
- On validation (fold 4), collect per-model 4-way softmax probabilities -> concat to 12-d features
- Train a multinomial Logistic Regression meta-learner on these 12-d features
- At test time, get per-model probs -> 12-d -> meta-learner -> final probs/preds

Assumptions / protocol
- Fixed 5-post bins
- Fold split: fold 0 = held-out internal test, fold 4 = validation, folds 1–3 = training
- Temporal features use a fixed FEATURE_KEYS_13 order
- Loss = 0.7 * Focal(α=1.0, γ=1.5) + 0.3 * class-weighted CE
- Early stopping on validation loss

Outputs
- bin_predictions_stacked_lr66.csv  (user_id, predicted_risk, prob_0..prob_3)
- stacker_lr.joblib (saved meta-learner)
- best_lstm_<key>.pth (base checkpoints)
- Console: evaluation for Validation (fold-4) and Internal Test (fold-0)
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.linear_model import LogisticRegression

# --------------------- Paths & Ensemble Spec ---------------------
TRAIN_CSV     = "data_with_instance_and_fold_labels.csv"
TEST_SDOH_CSV = "sdoh_evaluate_on_leaderboard.csv"

ENSEMBLE_SPECS = [
    {"key": "biobert_base",
     "train": "train_biobert_base_embeddings.npy",
     "test":  "test_biobert_base_embeddings.npy",
     "ckpt":  "best_lstm_biobert_base.pth"},
    {"key": "roberta_large",
     "train": "train_roberta_large_embeddings.npy",
     "test":  "test_roberta_large_embeddings.npy",
     "ckpt":  "best_lstm_roberta_large.pth"},
    {"key": "biomedbert",
     "train": "train_biomedbert_embeddings.npy",
     "test":  "test_biomedbert_embeddings.npy",
     "ckpt":  "best_lstm_biomedbert.pth"},
]

OUT_BIN_PRED = "bin_predictions_stacked_lr66.csv"
STACKER_PATH = "stacker_lr.joblib"
TMP_CKPT     = "TMPCKPT.pth"
SEED = 42

# --------------------- Utils ---------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def safe_read_csv(path, encodings=('utf-8', 'utf-8-sig', 'latin1', 'cp1252')):
    last_err = None
    for enc in encodings:
        try:
            print(f"[read_csv] Trying '{enc}' for {path}")
            df = pd.read_csv(path, encoding=enc)
            print(f"[read_csv] Success with '{enc}' ({len(df)} rows, {df.shape[1]} cols)")
            return df
        except Exception as e:
            last_err = e
    raise last_err

def compute_embed_norm(emb: np.ndarray):
    mu = emb.mean(axis=0).astype(np.float32)
    sigma = emb.std(axis=0).astype(np.float32)
    sigma[sigma < 1e-8] = 1e-8
    return mu, sigma

def apply_embed_norm(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray, l2: bool = True):
    x = (x - mu) / sigma
    if l2:
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n[n < 1e-8] = 1e-8
        x = x / n
    return x.astype(np.float32)

def load_embeddings_or_skip(path):
    if not os.path.exists(path):
        print(f"[SKIP] Missing embeddings: {path}")
        return None
    arr = np.load(path, allow_pickle=True)
    print(f"[emb] {path} shape = {arr.shape}")
    return arr

# --------------------- Loss ---------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=1.5, reduction='mean'):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.reduction = reduction
    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        alpha_t = self.alpha if isinstance(self.alpha, (float, int)) else self.alpha.gather(0, targets)
        loss = alpha_t * (1 - pt)**self.gamma * ce
        if self.reduction == 'mean': return loss.mean()
        if self.reduction == 'sum':  return loss.sum()
        return loss

# --------------------- Temporal features ---------------------
FEATURE_KEYS_13 = [
    'bin_avg_interval','bin_std_interval','bin_min_interval','bin_max_interval',
    'bin_last_interval','bin_running_avg_last3','bin_hour_entropy','bin_circadian_R',
    'bin_posts_per_day','bin_n_posts','bin_span_hours','delta_hours_since_prev_bin_end',
    'bin_index_norm'
]

class OptimizedSDOHProcessor:
    def __init__(self, max_posts_per_bin=5, n_bins=5):
        self.max_posts_per_bin = max_posts_per_bin
        self.n_bins = n_bins
        self.scaler = RobustScaler()
        self.all_sdoh_columns = [
            'CrisisAlcoholProblem_c','CrisisCivilLegal_c','CrisisCriminal_c',
            'CrisisEviction_c','CrisisDisasterExposure_c','CrisisFamilyRelationship_c',
            'CrisisFinancial_c','CrisisIntimatePartnerProblem_c','CrisisJob_c',
            'CrisisMentalHealth_c','CrisisOtherAddiction_c','CrisisRelationshipProblemOth_c',
            'CrisisSubstanceAbuse_c','CrisisPhysicalHealth_c','CrisisRecSuicideFriendFamily_c',
            'CrisisSchool_c'
        ]
        self.risk_labels = {0:'indicator',1:'ideation',2:'behavior',3:'attempt'}

    def load_and_prepare_train(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast labels, parse time, fill SDOH NA with 0, sort, and add original_index."""
        assert 'suicide_risk' in df.columns, "train df needs suicide_risk"
        df = df.copy()
        df['suicide_risk'] = df['suicide_risk'].astype(int)
        if 'post_created_utc' in df.columns:
            df['post_created_utc'] = pd.to_datetime(df['post_created_utc'], utc=True, errors='coerce')
        else:
            df['post_created_utc'] = pd.Timestamp.utcnow()
        for col in self.all_sdoh_columns:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(0.0)
        df = df.reset_index(drop=True)
        df['original_index'] = df.index
        df = df.sort_values(['user_id','post_created_utc','original_index'])
        return df

    def compute_temporal_features_from_times(self, timestamps: List[pd.Timestamp]) -> Dict[str, List[float]]:
        """Compute 13 per-step features; assume timestamps exist and are valid; pad to 5."""
        feats = {k: [] for k in FEATURE_KEYS_13}
        ts = sorted([t for t in timestamps if pd.notna(t)])
        if not ts:
            ts = [pd.Timestamp.utcnow()] * 5
        while len(ts) < 5:
            ts.append(ts[-1])
        # lightweight placeholders
        for bi in range(5):
            feats['bin_index_norm'].append(bi/4.0)
            feats['bin_n_posts'].append(1.0)
            for k in ['bin_avg_interval','bin_std_interval','bin_min_interval',
                      'bin_max_interval','bin_last_interval','bin_running_avg_last3']:
                feats[k].append(0.0)
            feats['bin_span_hours'].append(0.0)
            feats['bin_posts_per_day'].append(1.0)
            feats['delta_hours_since_prev_bin_end'].append(
                0.0 if bi == 0 else float((ts[bi] - ts[bi-1]).total_seconds()/3600.0)
            )
            feats['bin_hour_entropy'].append(0.0)
            feats['bin_circadian_R'].append(0.0)
        return feats

    def fit_temporal_scalers(self, all_features: List[Dict[str, List[float]]]):
        """Fit RobustScaler on flattened 5×13 features."""
        if not all_features:
            return
        rows = []
        for features in all_features:
            row = []
            for name in FEATURE_KEYS_13:
                vals = features.get(name, [])
                if len(vals) < 5:
                    vals = list(vals) + [0.0] * (5 - len(vals))
                row.extend(vals[:5])
            rows.append(row)
        self.scaler.fit(rows)

    def transform_temporal_features(self, features: Dict[str, List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        """Flatten to 65-d in FEATURE_KEYS_13 order and scale. Return zeros mask for API compatibility."""
        row = []
        for name in FEATURE_KEYS_13:
            vals = features[name]
            if len(vals) < 5:
                vals = list(vals) + [0.0] * (5 - len(vals))
            row.extend(vals[:5])
        scaled = self.scaler.transform([row])[0]
        missing_mask = np.zeros_like(scaled, dtype=np.float32)
        return scaled, missing_mask

# --------------------- Datasets ---------------------
class SimplifiedSDOHDataset(Dataset):
    """One sample = a 5-post bin."""
    def __init__(self, bin_indices: List[np.ndarray], df: pd.DataFrame,
                 processor: OptimizedSDOHProcessor, embeddings: np.ndarray,
                 emb_dim: int, is_training=True):
        self.bin_indices = bin_indices
        self.df = df
        self.processor = processor
        self.emb = embeddings
        self.emb_dim = emb_dim
        self.is_training = is_training
        self.items = self._build()

    def _build(self):
        data = []
        all_temporal_features = []
        print(f"Preparing SDOH data for {len(self.bin_indices)} bins...")
        for i, bin_idx in enumerate(self.bin_indices):
            if i % 200 == 0: print(f"Processing bin {i+1}/{len(self.bin_indices)}")
            bin_rows = self.df.loc[bin_idx].sort_values('post_created_utc')
            label = int(bin_rows['suicide_risk'].max())
            timestamps = pd.to_datetime(bin_rows['post_created_utc'], utc=True, errors='coerce').tolist()

            temporal_features = self.processor.compute_temporal_features_from_times(timestamps[:5])
            if self.is_training:
                all_temporal_features.append(temporal_features)

            roberta_sdoh_features = []
            for _, row in bin_rows.head(5).iterrows():
                rfeat = self.emb[row['original_index']]
                # SDOH NA -> 0
                sdoh = row[self.processor.all_sdoh_columns].astype(float).fillna(0.0).values \
                       if set(self.processor.all_sdoh_columns).issubset(row.index) \
                       else np.zeros(len(self.processor.all_sdoh_columns), dtype=np.float32)
                combined = np.concatenate([rfeat, sdoh.astype(np.float32)])
                roberta_sdoh_features.append(combined)

            while len(roberta_sdoh_features) < 5:
                roberta_sdoh_features.append(
                    roberta_sdoh_features[-1].copy() if roberta_sdoh_features
                    else np.zeros(self.emb_dim + len(self.processor.all_sdoh_columns), dtype=np.float32)
                )
            roberta_sdoh_features = roberta_sdoh_features[:5]

            data.append({
                'user_id': i,
                'label': label,
                'roberta_sdoh_features': np.array(roberta_sdoh_features, dtype=np.float32),
                'temporal_features': temporal_features,
                'time_gaps': np.array([float(j*24) for j in range(5)], dtype=np.float32),
            })

        if self.is_training and all_temporal_features:
            print(f"Fitting temporal scaler on {len(all_temporal_features)} bins...")
            self.processor.fit_temporal_scalers(all_temporal_features)
        print(f"Data prep complete. {len(data)} bins.")
        return data

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        scaled_temporal, missing_indicators = self.processor.transform_temporal_features(item['temporal_features'])
        return {
            'user_id': item['user_id'],
            'label': torch.tensor(item['label'], dtype=torch.long),
            'roberta_sdoh_features': torch.tensor(item['roberta_sdoh_features'], dtype=torch.float32),
            'temporal_features': torch.tensor(scaled_temporal.reshape(5, 13), dtype=torch.float32),
            'missing_indicators': torch.tensor(missing_indicators.reshape(5, 13), dtype=torch.float32),
            'time_gaps': torch.tensor(item['time_gaps'], dtype=torch.float32),
        }

class TestBinDataset(Dataset):
    """Test-time bins built from grouped indices."""
    def __init__(self, grouped_indices: List[np.ndarray], df: pd.DataFrame,
                 proc: OptimizedSDOHProcessor, emb_dim: int,
                 per_post_emb: np.ndarray = None, per_bin_emb: np.ndarray = None):
        self.groups = grouped_indices
        self.df = df
        self.proc = proc
        self.per_post_emb = per_post_emb
        self.per_bin_emb = per_bin_emb
        self.emb_dim = emb_dim
        if self.per_bin_emb is not None and len(self.per_bin_emb) < len(self.groups):
            print(f"[warn] per-bin embeddings ({len(self.per_bin_emb)}) < #bins ({len(self.groups)}); slicing bins.")
            self.groups = self.groups[:len(self.per_bin_emb)]

    def __len__(self): return len(self.groups)

    def __getitem__(self, i):
        idxs = self.groups[i]
        rows = self.df.loc[idxs].sort_values('post_created_utc')
        uid = int(rows['user_id'].iloc[0])
        ts = pd.to_datetime(rows['post_created_utc'], utc=True, errors='coerce').tolist()

        feats = self.proc.compute_temporal_features_from_times(ts)
        vals = []
        for k in FEATURE_KEYS_13:
            v = feats[k]
            if len(v) < 5: v = v + [0.0]*(5-len(v))
            vals.extend([float(x) for x in v[:5]])
        scaled = self.proc.scaler.transform([vals])[0].astype(np.float32)

        if self.per_post_emb is not None:
            rbt = self.per_post_emb[rows['original_index'].values].astype(np.float32)
        else:
            rbt = self.per_bin_emb[i].astype(np.float32)

        sd = rows[self.proc.all_sdoh_columns].astype(float).fillna(0.0).values
        roberta_sdoh = np.concatenate([rbt, sd], axis=1).astype(np.float32)

        time_gaps = np.array([float(j*24) for j in range(5)], dtype=np.float32)

        return {
            'user_id': uid,
            'label': torch.tensor(0, dtype=torch.long),
            'roberta_sdoh_features': torch.tensor(roberta_sdoh, dtype=torch.float32),
            'temporal_features': torch.tensor(scaled.reshape(5,13), dtype=torch.float32),
            'missing_indicators': torch.zeros(5,13, dtype=torch.float32),
            'time_gaps': torch.tensor(time_gaps, dtype=torch.float32),
        }

def collate(batch):
    return {
        'user_ids': [b['user_id'] for b in batch],
        'labels': torch.stack([b['label'] for b in batch]),
        'roberta_sdoh_features': torch.stack([b['roberta_sdoh_features'] for b in batch]),
        'temporal_features': torch.stack([b['temporal_features'] for b in batch]),
        'missing_indicators': torch.stack([b['missing_indicators'] for b in batch]),
        'time_gaps': torch.stack([b['time_gaps'] for b in batch]),
    }

# --------------------- Model ---------------------
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, x, mask=None):
        attn_weights = self.attention(x).squeeze(-1)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask, 0)
        pooled = torch.sum(attn_weights.unsqueeze(-1) * x, dim=1)
        return pooled

class OptimizedLSTMSDOHModel(nn.Module):
    def __init__(self, roberta_dim=768, sdoh_dim=16, temporal_dim=13,
                 hidden_dim=384, lstm_layers=2, dropout=0.2, num_classes=4):
        super().__init__()
        self.roberta_dim = roberta_dim
        self.sdoh_dim = sdoh_dim
        self.temporal_dim = temporal_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.roberta_proj = nn.Sequential(
            nn.Linear(roberta_dim, hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden_dim)
        )
        self.sdoh_proj = nn.Sequential(
            nn.Linear(sdoh_dim, hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden_dim)
        )

        temporal_feature_dim = 32
        self.temporal_encoder = nn.Sequential(
            nn.Linear(temporal_dim * 5, temporal_feature_dim),
            nn.BatchNorm1d(temporal_feature_dim),
            nn.GELU(), nn.Dropout(dropout)
        )

        self.bin_fusion = nn.Sequential(
            nn.Linear(hidden_dim + temporal_feature_dim, hidden_dim),
            nn.GELU(), nn.Dropout(dropout)
        )

        self.temporal_lstm = nn.LSTM(
            hidden_dim, hidden_dim, lstm_layers,
            batch_first=True, dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False
        )
        self.time_attention = AttentionPooling(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name: torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name: torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name: torch.nn.init.zeros_(param.data)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            torch.nn.init.zeros_(module.bias); torch.nn.init.ones_(module.weight)

    def forward(self, roberta_sdoh_features, temporal_features, missing_indicators, time_gaps):
        B, n_bins = roberta_sdoh_features.shape[:2]
        roberta_features = roberta_sdoh_features[:, :, :self.roberta_dim]
        sdoh_features = roberta_sdoh_features[:, :, self.roberta_dim:self.roberta_dim+self.sdoh_dim]

        roberta_proj = self.roberta_proj(roberta_features.view(-1, self.roberta_dim)).view(B, n_bins, -1)
        sdoh_proj    = self.sdoh_proj(sdoh_features.view(-1, self.sdoh_dim)).view(B, n_bins, -1)
        bin_text_features = (roberta_proj + sdoh_proj) / 2

        temporal_flat = temporal_features.view(B, -1)
        temporal_emb  = self.temporal_encoder(temporal_flat)
        temporal_emb_expanded = temporal_emb.unsqueeze(1).expand(-1, n_bins, -1)

        fused_bins = torch.cat([bin_text_features, temporal_emb_expanded], dim=-1)
        fused_bins = self.bin_fusion(fused_bins)

        lstm_out, _ = self.temporal_lstm(fused_bins)
        mask = torch.ones(B, n_bins, dtype=torch.bool, device=lstm_out.device)
        user_embed = self.time_attention(lstm_out, mask)
        logits = self.classifier(user_embed)
        return {'logits': logits, 'user_embedding': user_embed}

# --------------------- Training & Inference (base models) ---------------------
def combined_loss_fn(logits, targets, class_weights, focal_alpha=1.0, focal_gamma=1.5):
    focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    ce = nn.CrossEntropyLoss(weight=class_weights)
    return 0.7*focal(logits, targets) + 0.3*ce(logits, targets)

def train_lstm(train_loader, val_loader, device, emb_dim, num_epochs=80, lr=8e-5, weight_decay=1e-3, patience=10):
    model = OptimizedLSTMSDOHModel(roberta_dim=emb_dim, hidden_dim=384, lstm_layers=2,
                                   dropout=0.2, num_classes=4).to(device)
    # class weights from TRAIN set
    all_labels = [it['label'] for it in getattr(train_loader.dataset, "items", [])]
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    total_samples = len(all_labels)
    support_weights = total_samples / (len(unique_labels) * counts)
    risk_multipliers = np.array([1.0, 1.3, 2.3, 3.5], dtype=np.float32)
    class_weights = torch.tensor(support_weights * risk_multipliers, dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_loss = float('inf'); patience_ctr = 0

    for ep in range(1, num_epochs+1):
        model.train(); tr_loss=0.0; tr_p=[]; tr_y=[]
        for batch in train_loader:
            opt.zero_grad()
            out = model(batch['roberta_sdoh_features'].to(device),
                        batch['temporal_features'].to(device),
                        batch['missing_indicators'].to(device),
                        batch['time_gaps'].to(device))
            y = batch['labels'].to(device)
            loss = combined_loss_fn(out['logits'], y, class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tr_loss += loss.item()
            tr_p.extend(torch.argmax(out['logits'], dim=1).detach().cpu().numpy().tolist())
            tr_y.extend(y.detach().cpu().numpy().tolist())

        model.eval(); va_loss=0.0; va_p=[]; va_y=[]
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch['roberta_sdoh_features'].to(device),
                            batch['temporal_features'].to(device),
                            batch['missing_indicators'].to(device),
                            batch['time_gaps'].to(device))
                y = batch['labels'].to(device)
                loss = combined_loss_fn(out['logits'], y, class_weights)
                va_loss += loss.item()
                va_p.extend(torch.argmax(out['logits'], dim=1).detach().cpu().numpy().tolist())
                va_y.extend(y.detach().cpu().numpy().tolist())

        tr_wf1 = f1_score(tr_y, tr_p, average='weighted') if tr_y else 0.0
        va_wf1 = f1_score(va_y, va_p, average='weighted') if va_y else 0.0
        avg_tr_loss = tr_loss/max(1,len(train_loader))
        avg_va_loss = va_loss/max(1,len(val_loader))
        print(f"[LSTM] Epoch {ep:03d} | TrainLoss {avg_tr_loss:.4f} WF1 {tr_wf1:.4f} "
              f"| ValLoss {avg_va_loss:.4f} WF1 {va_wf1:.4f}")

        if avg_va_loss < best_val_loss:
            best_val_loss = avg_va_loss; patience_ctr = 0
            torch.save({'model_state_dict': model.state_dict(), 'best_val_loss': best_val_loss}, TMP_CKPT)
            print(f"[LSTM] New best checkpoint (ValLoss={best_val_loss:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"[LSTM] Early stop. Best Val Loss: {best_val_loss:.4f}")
                break

    ckpt = torch.load(TMP_CKPT, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"[LSTM] Loaded best model (Val Loss={ckpt.get('best_val_loss', -1):.4f})")
    return model

@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    uids, logits_all = [], []
    for batch in loader:
        out = model(batch['roberta_sdoh_features'].to(device),
                    batch['temporal_features'].to(device),
                    batch['missing_indicators'].to(device),
                    batch['time_gaps'].to(device))
        logits_all.append(out['logits'].detach().cpu().numpy())
        uids.extend(batch['user_ids'])
    return uids, np.vstack(logits_all)  # [N, num_classes]

@torch.no_grad()
def predict_probs(model, loader, device):
    uids, logits = predict_logits(model, loader, device)
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=1).numpy()
    return uids, probs

# --------------------- Helpers ---------------------
def make_bin_groups(df: pd.DataFrame) -> List[np.ndarray]:
    """Make non-overlapping 5-post bins within each user."""
    groups=[]
    for uid, g in df.groupby('user_id', sort=False):
        g = g.sort_values('post_created_utc')
        idx = g.index.to_list()
        for s in range(0, len(idx) - (len(idx) % 5), 5):
            groups.append(np.array(idx[s:s+5], dtype=int))
    return groups

def create_training_bins(df: pd.DataFrame) -> List[np.ndarray]:
    """Create non-overlapping 5-post bins across the sorted dataframe."""
    df = df.sort_values(['user_id', 'post_created_utc']).reset_index(drop=True)
    df['original_index'] = df.index
    bins = []
    for i in range(0, len(df) - 4, 5):
        bins.append(df.iloc[i:i+5].index.to_numpy())
    print(f"Created {len(bins)} training bins from {len(df)} rows")
    return bins

def assert_bins_single_fold(df: pd.DataFrame, bins: List[np.ndarray]):
    """Ensure each bin belongs to a single fold."""
    for b in bins:
        if df.loc[b, 'fold_label'].nunique() != 1:
            raise AssertionError("Bin spans multiple fold labels")

# --------------------- Meta-learner (Logistic Regression on probs) ---------------------
def train_meta_logreg(X_val: np.ndarray, y_val: np.ndarray) -> LogisticRegression:
    """Fit multinomial logistic regression on concatenated per-model probabilities."""
    meta = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        C=1.0,
        max_iter=200,
        n_jobs=None,
        random_state=SEED
    )
    meta.fit(X_val, y_val)
    joblib.dump(meta, STACKER_PATH)
    print(f"[Stacker] Saved meta-learner -> {STACKER_PATH}")
    return meta

def predict_meta_logreg(meta: LogisticRegression, X: np.ndarray) -> np.ndarray:
    """Return calibrated class probabilities [N, 4]."""
    return meta.predict_proba(X)

# --------------------- Pretty printer for eval ---------------------
def print_eval(name: str, y_true: np.ndarray, y_pred: np.ndarray):
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"\n===== {name} =====")
    print(f"Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f}\n")

    print("Per-class precision/recall/F1:")
    for c in range(len(pr)):
        print(f"  Class {c}: P={pr[c]:.4f} | R={rc[c]:.4f} | F1={f1[c]:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(np.unique(y_true))))
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)

    with np.errstate(invalid='ignore', divide='ignore'):
        row_sums = cm.sum(axis=1, keepdims=True)
        norm_cm = np.divide(cm, row_sums, where=row_sums!=0)
    print("\nRow-normalized confusion matrix (recall matrix):")
    print(np.round(norm_cm, 4))

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

# --------------------- Main ---------------------
def main():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Train CSV & processor
    df_train = safe_read_csv(TRAIN_CSV)
    proc = OptimizedSDOHProcessor(max_posts_per_bin=5, n_bins=5)
    dfp = proc.load_and_prepare_train(df_train)

    # Build bins & split by fold
    training_bins = create_training_bins(dfp)
    assert_bins_single_fold(dfp, training_bins)
    bin_folds = []
    for bin_idx in training_bins:
        g = dfp.loc[bin_idx]
        f = int(g['fold_label'].iloc[0])
        bin_folds.append(f)
    bins_by_fold = {k: [] for k in [0,1,2,3,4]}
    for b, f in zip(training_bins, bin_folds):
        bins_by_fold[f].append(b)
    test_bins = bins_by_fold[0]
    val_bins  = bins_by_fold[4]
    train_bins = bins_by_fold[1] + bins_by_fold[2] + bins_by_fold[3]
    print(f"Train bins: {len(train_bins)} | Val bins: {len(val_bins)} | Test bins: {len(test_bins)}")

    # Train one base model per embedding source + collect VAL probs
    trained_specs = []
    val_uid_ref = None
    val_prob_bank = []   # list of [N_val, 4], one per model
    y_val_ref = None

    for spec in ENSEMBLE_SPECS:
        train_path, test_path = spec["train"], spec["test"]
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"[SKIP] {spec['key']} missing train or test embeddings; skipping.")
            continue

        # Load & normalize train embeddings
        train_emb = load_embeddings_or_skip(train_path)
        assert train_emb is not None, "Train embeddings not found."
        assert len(train_emb) == len(dfp), f"Train emb rows ({len(train_emb)}) != train CSV rows ({len(dfp)})"
        mu, sigma = compute_embed_norm(train_emb)
        train_emb = apply_embed_norm(train_emb, mu, sigma, l2=True)
        EMB_DIM = int(train_emb.shape[1])
        print(f"[{spec['key']}] emb_dim = {EMB_DIM}")

        # Datasets / loaders (first dataset call fits temporal scaler)
        train_ds = SimplifiedSDOHDataset(train_bins, dfp, proc, train_emb, emb_dim=EMB_DIM, is_training=True)
        val_ds   = SimplifiedSDOHDataset(val_bins,   dfp, proc, train_emb, emb_dim=EMB_DIM, is_training=False)
        train_loader = DataLoader(train_ds, batch_size=12, shuffle=True,  collate_fn=collate,
                                  pin_memory=torch.cuda.is_available(), num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, collate_fn=collate,
                                  pin_memory=torch.cuda.is_available(), num_workers=0)

        # Train & save checkpoint
        model = train_lstm(train_loader, val_loader, device=device, emb_dim=EMB_DIM,
                           num_epochs=80, lr=8e-5, weight_decay=1e-3, patience=10)
        torch.save({'model_state_dict': model.state_dict()}, spec["ckpt"])
        print(f"[{spec['key']}] Saved checkpoint -> {spec['ckpt']}")

        # Collect validation probs for meta-learner
        uids_val, probs_val = predict_probs(model, val_loader, device=device)  # [N_val, 4]
        if val_uid_ref is None:
            val_uid_ref = uids_val
            y_val_ref = np.array([it['label'] for it in val_ds.items], dtype=np.int64)
        else:
            assert uids_val == val_uid_ref, "[VAL] user/bin ordering mismatch across models."
        val_prob_bank.append(probs_val)

        # Record for test-time inference
        spec["_mu"], spec["_sigma"], spec["_emb_dim"] = mu, sigma, EMB_DIM
        trained_specs.append(spec)

    if not trained_specs:
        raise RuntimeError("No backbones trained (missing files?).")

    # Train meta-learner on validation (stacking on probs)
    X_val = np.concatenate(val_prob_bank, axis=1)  # (num_models*4 = 12)
    meta = train_meta_logreg(X_val, y_val_ref)

    # === Validation (fold-4) evaluation of the stacked ensemble ===
    val_probs_final = predict_meta_logreg(meta, X_val)
    val_preds_final = val_probs_final.argmax(axis=1)
    print_eval("Validation (fold-4) — stacked ensemble", y_val_ref, val_preds_final)

    # === Internal test (fold-0) evaluation of the stacked ensemble ===
    test0_bins = bins_by_fold[0]
    if len(test0_bins) > 0:
        test0_prob_bank = []
        test0_uid_ref = None
        y_test0 = None

        for spec in trained_specs:
            train_path = spec["train"]
            train_emb_full = load_embeddings_or_skip(train_path)
            mu, sigma = spec["_mu"], spec["_sigma"]
            EMB_DIM = spec["_emb_dim"]
            train_emb_full = apply_embed_norm(train_emb_full, mu, sigma, l2=True)

            test0_ds = SimplifiedSDOHDataset(test0_bins, dfp, proc, train_emb_full, emb_dim=EMB_DIM, is_training=False)
            test0_loader = DataLoader(test0_ds, batch_size=128, shuffle=False, collate_fn=collate)

            model = OptimizedLSTMSDOHModel(roberta_dim=EMB_DIM, hidden_dim=384, lstm_layers=2, dropout=0.1, num_classes=4).to(device)
            state = torch.load(spec["ckpt"], map_location=device)
            model.load_state_dict(state['model_state_dict'])
            uids_test0, probs_test0 = predict_probs(model, test0_loader, device=device)

            if test0_uid_ref is None:
                test0_uid_ref = uids_test0
                y_test0 = np.array([it['label'] for it in test0_ds.items], dtype=np.int64)
            else:
                assert test0_uid_ref == uids_test0, "[Fold-0] user/bin ordering mismatch across models."

            test0_prob_bank.append(probs_test0)

        if len(test0_prob_bank) > 0:
            X_test0 = np.concatenate(test0_prob_bank, axis=1)  # shape [N0, 12]
            test0_probs_final = predict_meta_logreg(meta, X_test0)
            test0_preds_final = test0_probs_final.argmax(axis=1)
            print_eval("Internal Test (fold-0) — stacked ensemble", y_test0, test0_preds_final)
    else:
        print("[INFO] No fold-0 bins available; skipping internal test evaluation.")

    # ===== External test (leaderboard) inference =====
    df_test = safe_read_csv(TEST_SDOH_CSV)
    assert 'user_id' in df_test.columns and 'post_created_utc' in df_test.columns
    df_test['post_created_utc'] = pd.to_datetime(df_test['post_created_utc'], utc=True, errors='coerce')
    df_test = df_test.reset_index(drop=True)
    df_test['original_index'] = df_test.index
    groups = make_bin_groups(df_test)
    print(f"[test] formed {len(groups)} bins (5 rows each) from {len(df_test)} rows.")

    test_uid_ref = None
    test_prob_bank = []

    for spec in trained_specs:
        print(f"[Test] Inference with {spec['key']}")
        arr = load_embeddings_or_skip(spec["test"])
        assert arr is not None, f"Missing test embeddings for {spec['key']}"
        EMB_DIM = spec["_emb_dim"]
        mu, sigma = spec["_mu"], spec["_sigma"]

        per_post_emb = None; per_bin_emb = None
        if arr.ndim == 2 and arr.shape[1] == EMB_DIM:
            if arr.shape[0] < len(df_test):
                raise AssertionError(f"Per-post embeddings rows ({arr.shape[0]}) < test rows ({len(df_test)}).")
            arr = apply_embed_norm(arr, mu, sigma, l2=True); per_post_emb = arr
        elif arr.ndim == 3 and arr.shape[1] == 5 and arr.shape[2] == EMB_DIM:
            flat = arr.reshape(-1, EMB_DIM)
            flat = apply_embed_norm(flat, mu, sigma, l2=True)
            per_bin_emb = flat.reshape(arr.shape[0], 5, EMB_DIM).astype(np.float32)
            if per_bin_emb.shape[0] > len(groups):
                print(f"[warn] per-bin embeddings ({per_bin_emb.shape[0]}) > #bins ({len(groups)}); slicing embeddings.")
                per_bin_emb = per_bin_emb[:len(groups)]
        else:
            raise AssertionError(f"[{spec['key']}] Unsupported test embeddings shape: {arr.shape} (expected (*,{EMB_DIM}) or (*,5,{EMB_DIM}))")

        test_proc = OptimizedSDOHProcessor(max_posts_per_bin=5, n_bins=5)
        test_proc.scaler = proc.scaler
        test_proc.all_sdoh_columns = proc.all_sdoh_columns
        test_proc.risk_labels = proc.risk_labels

        test_ds = TestBinDataset(groups, df_test, test_proc, emb_dim=EMB_DIM,
                                 per_post_emb=per_post_emb, per_bin_emb=per_bin_emb)
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate)

        model = OptimizedLSTMSDOHModel(roberta_dim=EMB_DIM, hidden_dim=384, lstm_layers=2, dropout=0.1, num_classes=4).to(device)
        state = torch.load(spec["ckpt"], map_location=device)
        model.load_state_dict(state['model_state_dict'])
        uids_test, probs_test = predict_probs(model, test_loader, device=device)

        if test_uid_ref is None:
            test_uid_ref = uids_test
        else:
            assert test_uid_ref == uids_test, "[TEST] user/bin ordering mismatch across models."

        test_prob_bank.append(probs_test)

    X_test = np.concatenate(test_prob_bank, axis=1)  # [N_test, 12]
    if os.path.exists(STACKER_PATH):
        meta = joblib.load(STACKER_PATH)
        print(f"[Stacker] Loaded meta-learner from {STACKER_PATH}")
    probs_final = predict_meta_logreg(meta, X_test)
    preds_final = probs_final.argmax(axis=1)

    out = pd.DataFrame({'user_id': test_uid_ref, 'predicted_risk': preds_final})
    for c in range(probs_final.shape[1]):
        out[f'prob_{c}'] = probs_final[:, c]
    out.to_csv(OUT_BIN_PRED, index=False)
    print(f"[OUT] Wrote stacked predictions to {OUT_BIN_PRED}")
    print(out['predicted_risk'].value_counts().sort_index())

if __name__ == "__main__":
    main()
