import torch, pandas as pd, numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

# ---------- Data ----------
class InferenceGroupedDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.G = len(df)//5
    def __len__(self): return self.G
    def __getitem__(self, idx):
        s, e = idx*5, idx*5+5
        return self.df.iloc[s:e]["text"].tolist()

class Collator:
    def __init__(self, tok, max_len=256):
        self.tok, self.max_len = tok, max_len
    def __call__(self, batch):
        flat, sizes = [], []
        for texts in batch:
            flat.extend(texts); sizes.append(len(texts))
        enc = self.tok(flat, padding=True, truncation=True,
                       max_length=self.max_len, return_tensors="pt")
        return enc, torch.tensor(sizes)

# ---------- Model ----------
class RobertaMeanPoolClassifier(nn.Module):
    def __init__(self, roberta_name="roberta-base", num_classes=4):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(roberta_name)
        hid = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hid,128), nn.BatchNorm1d(128), nn.GELU(),
            nn.Linear(128,16), nn.BatchNorm1d(16), nn.GELU(),
            nn.Linear(16,num_classes)
        )
    @torch.no_grad()
    def _mean_pool_groups(self, cls_batch, sizes):
        out, off = [], 0
        for k in sizes.tolist():
            out.append(cls_batch[off:off+k].mean(dim=0, keepdim=True)); off+=k
        return torch.cat(out,0)
    def forward(self, enc, sizes):
        out = self.backbone(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        cls = out.last_hidden_state[:,0,:]
        pooled = self._mean_pool_groups(cls,sizes)
        return self.head(pooled)

# ---------- Inference function ----------
def predict_labels(csv_path="sdoh_evaluate_on_leaderboard.csv",
                   ckpt="Biomed_meanpool_best.pt",
                   roberta_name="roberta-base", num_classes=4,
                   batch_size=16, max_len=256, device=None):
    if device is None: device="cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"post_sequence":"text"})
    assert len(df)%5==0, "Number of rows is not a multiple of 5"
    tok = AutoTokenizer.from_pretrained(roberta_name)
    ds = InferenceGroupedDataset(df)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    collate_fn=Collator(tok,max_len))
    model = RobertaMeanPoolClassifier(roberta_name,num_classes).to(device)
    state = torch.load(ckpt,map_location=device)
    model.load_state_dict(state if isinstance(state,dict) else state["state_dict"])
    model.eval()
    preds=[]
    with torch.no_grad():
        for enc,sizes in dl:
            enc={k:v.to(device) for k,v in enc.items()}
            sizes=sizes.to(device)
            logits=model(enc,sizes)
            preds.extend(logits.argmax(1).cpu().numpy().tolist())
    return np.array(preds)

if __name__ == "__main__":
    # roberta_name="/prj0129/jzh4027/IEEE/local_models/models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract/snapshots/d673b8835373c6fa116d6d8006b33d48734e305d"
    roberta_name = "roberta-base"
    ckpt = "df3_roberta.pt"
    preds = predict_labels("sdoh_evaluate_on_leaderboard.csv", ckpt = ckpt,roberta_name=roberta_name)
    print("Number of predicted labels:", len(preds))
    print("First 20 predictions:", preds[:20])
    # Save as npy file
    np.save("df3_roberta.npy", preds)
    print("Predictions saved to df3_roberta.npy")
