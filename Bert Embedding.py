# pip install -U transformers torch tqdm numpy pandas

import os, gc
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# Silence tokenizer parallelism warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ========= Config =========
DATA_PATH       = "sdoh_evaluate_on_leaderboard.csv" #or "data_with_instance_and_fold_labels.csv"
# Use BioBERT (base). For large: "dmis-lab/biobert-large-cased-v1.1"
MODEL_NAME      = "model name"
MAX_LENGTH      = 512
INIT_BATCH_SIZE = 16    # base is lighter; 16 is usually fine (lower if OOM)
MIN_BATCH_SIZE  = 1

OUT_LONG_CSV = "test_biobert_base_embeddings.csv"
OUT_NPY      = "test_biobert_base_embeddings.npy"
OUT_ROW_META = "test_row_meta.csv"
COL_PREFIX   = "biobert_"   # optional

# ========= Pooling & Encoding =========
@torch.no_grad()
def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mask-aware mean pooling -> [B, H]"""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts

def encode_batch_texts(texts, tokenizer, model, device, use_cuda, max_length=256):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        pad_to_multiple_of=8 if use_cuda else None,  # small throughput win on GPU
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    if use_cuda:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(**enc)
    else:
        outputs = model(**enc)
    pooled = mean_pool(outputs.last_hidden_state, enc["attention_mask"])
    # free memory
    del outputs, enc
    if use_cuda:
        torch.cuda.empty_cache()
    gc.collect()
    return pooled.cpu().numpy()

def main():
    # ========= Load CSV (row-wise) =========
    df = pd.read_csv(DATA_PATH)
    if "post_sequence" not in df.columns:
        raise ValueError("CSV must contain the targeted text column 'post_sequence'.")

    # Keep original row index for traceability
    df = df.reset_index().rename(columns={"index": "original_row_index"})
    # Normalize text
    df["post_sequence"] = df["post_sequence"].fillna("").astype(str)

    # Optional: ensure a timestamp column exists for metadata (not required)
    if "post_created_utc" not in df.columns:
        df["post_created_utc"] = pd.NaT

    num_rows = len(df)
    print(f"Loaded {num_rows} rows. Embedding every row (ignoring user_id).")

    # ========= Model =========
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device:", device)

    # For BERT-family (BioBERT), no add_prefix_space is needed
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    dtype = torch.float16 if use_cuda else torch.float32
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    hidden_size = getattr(model.config, "hidden_size", None)
    assert hidden_size is not None, "Could not determine hidden size from model.config.hidden_size"
    print(f"Model: {MODEL_NAME} | hidden_size={hidden_size}")

    # ========= Prepare long table =========
    meta_cols = [c for c in df.columns if c != "post_sequence"]
    long_df = pd.DataFrame({
        "row_id": np.arange(num_rows),
        "text": df["post_sequence"].tolist()
    })
    long_df = pd.concat([long_df, df[meta_cols].reset_index(drop=True)], axis=1)

    # ========= Encode with adaptive batching =========
    if os.path.exists(OUT_LONG_CSV):
        os.remove(OUT_LONG_CSV)

    cur_bs = INIT_BATCH_SIZE
    WRITE_HEADER = True
    all_feats = np.zeros((num_rows, hidden_size), dtype=np.float32)

    pbar = tqdm(total=num_rows, desc=f"Encoding with {MODEL_NAME.split('/')[-1]}")
    start = 0
    while start < num_rows:
        end_try = min(num_rows, start + cur_bs)
        batch_texts = long_df.loc[start:end_try-1, "text"].tolist()
        try:
            batch_emb = encode_batch_texts(batch_texts, tokenizer, model, device, use_cuda, MAX_LENGTH)

            # write long-form CSV chunk: [meta + embeddings]
            chunk_meta = long_df.loc[start:end_try-1, ["row_id"] + meta_cols].reset_index(drop=True)
            emb_df = pd.DataFrame(batch_emb, columns=[f"{COL_PREFIX}{i}" for i in range(batch_emb.shape[1])])
            out_chunk = pd.concat([chunk_meta, emb_df], axis=1)
            out_chunk.to_csv(OUT_LONG_CSV, index=False, mode="a", header=WRITE_HEADER)
            WRITE_HEADER = False

            # fill compact array
            all_feats[start:end_try, :] = batch_emb

            start = end_try
            pbar.update(len(batch_emb))

            # gentle auto-increase if previously shrunk
            if cur_bs < INIT_BATCH_SIZE:
                cur_bs = min(INIT_BATCH_SIZE, cur_bs * 2)

        except RuntimeError as e:
            msg = str(e).lower()
            if ("out of memory" in msg or "cuda" in msg) and cur_bs > MIN_BATCH_SIZE:
                if use_cuda:
                    torch.cuda.empty_cache()
                gc.collect()
                cur_bs = max(MIN_BATCH_SIZE, cur_bs // 2)
                print(f"[WARN] OOM: reduce batch size to {cur_bs} and retry...")
            else:
                raise

    pbar.close()

    # ========= Save compact tensor + row meta =========
    np.save(OUT_NPY, all_feats)  # shape [num_rows, hidden_size]

    row_meta = df[["original_row_index"]].copy()
    row_meta["row_id"] = np.arange(num_rows)
    row_meta = row_meta[["row_id", "original_row_index"]]
    row_meta.to_csv(OUT_ROW_META, index=False)

    print(f"Saved long CSV: {OUT_LONG_CSV}")
    print(f"Saved npy tensor: {OUT_NPY}  (shape={all_feats.shape})")
    print(f"Saved row meta: {OUT_ROW_META}")

if __name__ == "__main__":
    main()
