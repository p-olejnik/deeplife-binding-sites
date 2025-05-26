#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score

# Data paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
EMB_DIR = ROOT_DIR / "embeddings"
EMB_DIR.mkdir(exist_ok=True)

# Load CSVs
seq_df = pd.read_csv(DATA_DIR / "sequences.csv", index_col=0)
if "sequence" not in seq_df.columns:
    seq_df = seq_df.rename(columns={seq_df.columns[0]: "sequence"})
pddt_df = pd.read_csv(DATA_DIR / "pockets_mean_plddt.csv")
rmsd_df = pd.read_csv(DATA_DIR / "pockets_rmsd.csv")
hack_df = pd.read_csv(DATA_DIR / "hackathon-data(in).csv")

# Merge and label
metrics = (
    hack_df
    .merge(pddt_df, on=["uniprot_id", "alphafold_pocket_selection"]) 
    .merge(rmsd_df, on=["uniprot_id", "alphafold_pocket_selection"]))
metrics["is_cryptic"] = ((metrics["mean_plddt"] < 70.0) | (metrics["pocket_rmsd"] > 2.0)).astype(int)
metrics["alphafold_id"] = (
    metrics["alphafold_pocket_selection"]
    .str.split(" and", n=1)
    .str[0]
    .str.strip()
)

# Sanity-check, fetch missing sequences
missing = set(metrics["alphafold_id"]) - set(seq_df.index)
if missing:
    print(f"Fetching {len(missing)} missing sequences from AlphaFold DB with 20 threads...")
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def fetch_seq(af_id):
        uniprot = af_id.split('-')[1]
        url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot}"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return af_id, None, f"HTTP {resp.status_code}"
        data = resp.json()
        if isinstance(data, list) and data and "uniprotSequence" in data[0]:
            return af_id, data[0]["uniprotSequence"], None
        return af_id, None, "Unexpected format"

    total = len(missing)
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_id = {executor.submit(fetch_seq, af_id): af_id for af_id in missing}
        for count, future in enumerate(as_completed(future_to_id), start=1):
            af_id = future_to_id[future]
            try:
                _, seq, err = future.result()
            except Exception as e:
                seq, err = None, str(e)
            if seq:
                seq_df.loc[af_id] = seq
                print(f"[{count}/{total}] Fetched {af_id}")
            else:
                print(f"[{count}/{total}] Failed {af_id}: {err}")
    seq_df.to_csv(DATA_DIR / "sequences.csv")
    print(f"Appended {total} sequences and updated sequences.csv")

# Drop any still-missing
failed = [af for af in missing if af not in seq_df.index]
if failed:
    print(f"Warning: dropping {len(failed)} pockets due to missing sequences: {failed}")
    metrics = metrics[~metrics["alphafold_id"].isin(failed)]

# Split into train/test
train_df, test_df = train_test_split(
    metrics,
    test_size=0.2,
    stratify=metrics["is_cryptic"],
    random_state=42,
)

# Main block - load model, embed, train
if __name__ == "__main__":
    # Load ESM-2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = torch.hub.load(
        "facebookresearch/esm:main",
        "esm2_t33_650M_UR50D"
    )
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    # Embedding functions
    def embed_sequence(name: str, seq: str) -> torch.Tensor:
        max_len = 1022
        chunks = []
        for i in range(0, len(seq), max_len):
            sub = seq[i : i + max_len]
            data = [(name, sub)]
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(device)
            with torch.no_grad():
                out = model(tokens, repr_layers=[model.num_layers])
            rep = out["representations"][model.num_layers][0, 1:-1].cpu().numpy()
            chunks.append(rep)
        arr = np.concatenate(chunks, axis=0)
        return torch.from_numpy(arr.mean(axis=0))

    def embed_fn(seq_id: str) -> torch.Tensor:
        cache_path = EMB_DIR / f"{seq_id}.pt"
        if cache_path.exists():
            return torch.load(cache_path)
        seq = seq_df.loc[seq_id, "sequence"]
        emb = embed_sequence(seq_id, seq)
        torch.save(emb, cache_path)
        return emb

    # Dataset, DataLoader
    class PocketDataset(Dataset):
        def __init__(self, df, embed_fn, seq_index):
            self.df = df.reset_index(drop=True)
            self.embed_fn = embed_fn
            self.seq_index = set(seq_index)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.loc[idx]
            raw_pref = row["alphafold_id"]
            if raw_pref in self.seq_index:
                seq_id = raw_pref
            else:
                cands = [x for x in self.seq_index if raw_pref in x or x in raw_pref]
                if len(cands) == 1:
                    seq_id = cands[0]
                else:
                    raise KeyError(
                        f"Cannot resolve '{raw_pref}' to unique sequence (candidates: {cands})"
                    )
            emb = self.embed_fn(seq_id)
            label = torch.tensor(row["is_cryptic"], dtype=torch.float32)
            return emb, label

    train_loader = DataLoader(
        PocketDataset(train_df, embed_fn, seq_df.index),
        batch_size=32, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        PocketDataset(test_df, embed_fn, seq_df.index),
        batch_size=64, shuffle=False, num_workers=0
    )

    # Model head and training utils
    class PLM_nn(nn.Module):
        def __init__(self, input_dim, layer_width=100, dropout=0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, layer_width),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(layer_width, layer_width),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(layer_width, 1)
            )

        def forward(self, x):
            return self.net(x)

    model_clf = PLM_nn(
        input_dim=model.embed_dim,
        layer_width=256,
        dropout=0.3
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model_clf.parameters(), lr=1e-4)

    def evaluate(loader):
        model_clf.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for emb, label in loader:
                emb = emb.to(device)
                logits = model_clf(emb).cpu().squeeze(-1)
                all_logits.append(logits)
                all_labels.append(label)
        logits = torch.cat(all_logits).numpy()
        labels = torch.cat(all_labels).numpy()
        probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
        return {
            "ROC-AUC": roc_auc_score(labels, probs),
            "Accuracy@0.5": accuracy_score(labels, probs > 0.5)
        }

    # Training loop with progress
    EPOCHS = 20
    print(f"Starting training for {EPOCHS} epochs...")
    total_batches = len(train_loader)
    best_auc = 0.0
    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
        model_clf.train()
        loss_sum = 0.0
        for batch_idx, (emb, label) in enumerate(train_loader, start=1):
            emb, label = emb.to(device), label.to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits = model_clf(emb)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * emb.size(0)
            if batch_idx % 10 == 0 or batch_idx == total_batches:
                print(f"[Batch {batch_idx}/{total_batches}] batch_loss: {loss.item():.4f}")
        avg_loss = loss_sum / len(train_loader.dataset)
        metrics_eval = evaluate(test_loader)
        auc = metrics_eval["ROC-AUC"]
        print(
            f"Epoch {epoch} complete. Avg loss: {avg_loss:.4f}. "
            f"Test AUC: {auc:.4f}"
        )
        if auc > best_auc:
            best_auc = auc
            torch.save(
                model_clf.state_dict(), ROOT_DIR / "best_plm_nn.pt"
            )
            print("Saved new best model")
    final_metrics = evaluate(test_loader)
    print(f"\nFinal Test Metrics: {final_metrics}")
