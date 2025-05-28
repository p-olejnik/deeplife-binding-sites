#!/usr/bin/env python
from __future__ import annotations

import re
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# ──────────────────────────────────────────────────────────────────────────────
# Globals and seeds
# ──────────────────────────────────────────────────────────────────────────────
torch.manual_seed(42)

# Lazily-initialized caches (per process)
_ESM_MODEL  = None
_ESM_ALPH   = None
_BATCH_CONV = None
_ESM_DEVICE = None

# ──────────────────────────────────────────────────────────────────────────────
# Top-level helpers (picklable)
# ──────────────────────────────────────────────────────────────────────────────
def parse_indices(sel: str) -> List[int]:
    """Convert a 'resi 12+34+56' style selection into 0-based indices."""
    m = re.search(r"resi\s*([0-9+]+)", sel)
    return [int(x) - 1 for x in m.group(1).split("+")] if m else []

def pad_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    embs = [b[0] for b in batch]
    lbls = [b[1] for b in batch]
    return (
        torch.nn.utils.rnn.pad_sequence(embs, batch_first=True),
        torch.nn.utils.rnn.pad_sequence(lbls, batch_first=True),
    )

@torch.no_grad()
def embed_seq(name: str,
              seq: str,
              max_len: int = 1022) -> torch.Tensor:
    """
    Embed a protein sequence with ESM-2.

    The ESM model is loaded on the first call per process and cached
    in globals so DataLoader workers reuse it.
    """
    global _ESM_MODEL, _ESM_ALPH, _BATCH_CONV, _ESM_DEVICE

    if _ESM_MODEL is None:
        _ESM_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _ESM_MODEL, _ESM_ALPH = torch.hub.load(
            "facebookresearch/esm:main", "esm2_t33_650M_UR50D"
        )
        _ESM_MODEL.eval().to(_ESM_DEVICE)
        _BATCH_CONV = _ESM_ALPH.get_batch_converter()

    chunks: List[torch.Tensor] = []
    for i in range(0, len(seq), max_len):
        sub = seq[i : i + max_len]
        _, _, tokens = _BATCH_CONV([(name, sub)])
        tokens = tokens.to(_ESM_DEVICE)
        out = _ESM_MODEL(tokens, repr_layers=[_ESM_MODEL.num_layers])
        rep = out["representations"][_ESM_MODEL.num_layers][0, 1:-1]
        chunks.append(rep.cpu()) # keep CPU tensors to save GPU memory
    return torch.cat(chunks, dim=0)

# ──────────────────────────────────────────────────────────────────────────────
# Dataset and model classes
# ──────────────────────────────────────────────────────────────────────────────
class PocketResidueDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 seqdf: pd.DataFrame,
                 embed_fn):
        self.df       = df.reset_index(drop=True)
        self.seqdf    = seqdf
        self.embed_fn = embed_fn

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.loc[i]
        af  = row["alphafold_id"]
        seq = self.seqdf.at[af, "sequence"]
        emb = self.embed_fn(af, seq) # [L, D]
        L   = emb.size(0)
        lbl = torch.zeros(L)
        for idx in parse_indices(row["pdb_pocket_selection"]):
            if 0 <= idx < L:
                lbl[idx] = 1
        return emb, lbl

class Predictor(nn.Module):
    def __init__(self,
                 D: int,
                 H: int = 128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(D, H),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(H, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.size()
        y = self.fc(x.view(B * L, D))
        return y.view(B, L)

# ──────────────────────────────────────────────────────────────────────────────
# Data-prep function (runs once in parent)
# ──────────────────────────────────────────────────────────────────────────────
def prepare_data():
    ROOT_DIR  = Path(__file__).resolve().parent
    DATA_DIR  = ROOT_DIR / "data"
    EMB_DIR   = ROOT_DIR / "embeddings"; EMB_DIR.mkdir(exist_ok=True)
    MODEL_DIR = ROOT_DIR / "models";     MODEL_DIR.mkdir(exist_ok=True)

    # Load CSVs
    seq_df   = pd.read_csv(DATA_DIR / "sequences.csv", index_col=0)
    if "sequence" not in seq_df.columns:
        seq_df = seq_df.rename(columns={seq_df.columns[0]: "sequence"})

    pddt_df  = pd.read_csv(DATA_DIR / "pockets_mean_plddt.csv")
    rmsd_df  = pd.read_csv(DATA_DIR / "pockets_rmsd.csv")
    hack_df  = pd.read_csv(DATA_DIR / "hackathon-data(in).csv")

    # Merge and clean pocket selections
    metrics = (
        hack_df
        .merge(pddt_df, on=["uniprot_id", "alphafold_pocket_selection"])
        .merge(rmsd_df, on=["uniprot_id", "alphafold_pocket_selection"])
    )
    if "pdb_pocket_selection_x" in metrics.columns:
        metrics = metrics.rename(columns={"pdb_pocket_selection_x": "pdb_pocket_selection"})
    elif "pdb_pocket_selection" not in metrics.columns:
        raise KeyError("pdb_pocket_selection missing after merge!")
    metrics = metrics.drop(
        columns=[c for c in ("pdb_pocket_selection_y",) if c in metrics]
    )

    # Label cryptic pockets
    metrics["is_cryptic"] = (
        (metrics["mean_plddt"] < 70.0) |
        (metrics["pocket_rmsd"] > 2.0)
    ).astype(int)

    # Extract AF ID prefix
    metrics["alphafold_id"] = (
        metrics["alphafold_pocket_selection"]
        .str.split(" and", n=1)
        .str[0]
        .str.strip()
    )

    # Fetch missing sequences exactly once
    missing = set(metrics["alphafold_id"]) - set(seq_df.index)
    if missing:
        print(f"Fetching {len(missing)} missing sequences…")

        def _fetch_seq(af_id: str) -> Tuple[str, str | None]:
            up = af_id.split("-", 1)[1]
            url = f"https://alphafold.ebi.ac.uk/api/prediction/{up}"
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    return af_id, data[0].get("uniprotSequence")
            except Exception:
                pass
            return af_id, None

        with ThreadPoolExecutor(max_workers=20) as exe:
            for idx, (af_id, seq) in enumerate(exe.map(_fetch_seq, missing), start=1):
                if seq:
                    seq_df.loc[af_id] = seq
                    print(f"[{idx}/{len(missing)}] Fetched {af_id}")
                else:
                    print(f"[{idx}/{len(missing)}] Failed  {af_id}")

        seq_df.to_csv(DATA_DIR / "sequences.csv")
        still_missing = [x for x in missing if x not in seq_df.index]
        if still_missing:
            print(f"Dropping {len(still_missing)} pockets: {still_missing}")
            metrics = metrics[~metrics["alphafold_id"].isin(still_missing)]

    # Train/test split and subsample
    train_df, test_df = train_test_split(
        metrics,
        test_size=0.2,
        stratify=metrics["is_cryptic"],
        random_state=42
    )
    train_df = train_df.sample(frac=0.25, random_state=42).reset_index(drop=True)

    return train_df, test_df, seq_df, MODEL_DIR

# ──────────────────────────────────────────────────────────────────────────────
# Main train / eval
# ──────────────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, test_df, seq_df, MODEL_DIR = prepare_data()

    # Build datasets and loaders
    train_ds = PocketResidueDataset(train_df, seq_df, embed_seq)
    test_ds  = PocketResidueDataset(test_df,  seq_df, embed_seq)

    BATCH_SIZE = 16
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=pad_collate,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=pad_collate,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available(),
    )

    # Classifier and optimizer
    # get emb dim by embedding a single residue 'A'
    D = embed_seq("dummy", "A").size(-1)
    clf  = Predictor(D).to(device)
    crit = nn.BCEWithLogitsLoss()
    opt  = torch.optim.Adam(clf.parameters(), lr=1e-4)

    # Training loop
    EPOCHS = 5
    for ep in range(1, EPOCHS + 1):
        clf.train()
        print(f"\n=== Epoch {ep}/{EPOCHS} ===")
        for bidx, (emb, lbl) in enumerate(train_loader, 1):
            emb, lbl = emb.to(device), lbl.to(device)
            opt.zero_grad()
            logits = clf(emb)
            loss   = crit(logits, lbl)
            loss.backward()
            opt.step()
            if bidx % 10 == 0 or bidx == len(train_loader):
                print(f"[{bidx}/{len(train_loader)}] loss: {loss:.4f}")

        ckpt = MODEL_DIR / f"clf_epoch{ep}.pth"
        torch.save(clf.state_dict(), ckpt)
        print(f"Saved checkpoint {ckpt.name}")

    # Final checkpoint
    final_ckpt = MODEL_DIR / "clf_final.pth"
    torch.save(clf.state_dict(), final_ckpt)
    print(f"Saved final model: {final_ckpt.name}")

    # Evaluation
    clf.eval()
    ys, ts = [], []
    with torch.no_grad():
        for emb, lbl in test_loader:
            emb = emb.to(device)
            logits = clf(emb)
            preds  = torch.sigmoid(logits).cpu().numpy().ravel()
            trues  = lbl.numpy().ravel()
            ys.append(preds); ts.append(trues)

    y = np.concatenate(ys)
    t = np.concatenate(ts)
    print("Final ROC-AUC:", roc_auc_score(t, y))
    print("Final Acc@0.5:", accuracy_score(t, y > 0.5))

# ──────────────────────────────────────────────────────────────────────────────
# Entry point  (executed by the parent process only)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Explicit is better than implicit on Windows
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
