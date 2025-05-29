#!/usr/bin/env python
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score
from concurrent.futures import ThreadPoolExecutor
import requests
from Trans_encoder import Full_Attention, Attention_Layer, FeedForward, Encoder_layer
from torch.amp import autocast, GradScaler

#scaler = GradScaler()

torch.manual_seed(42)

# Paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
EMB_DIR  = ROOT_DIR / "embeddings_3"
MODEL_DIR = ROOT_DIR / "models"
EMB_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Save utility
def save_model(model, path):
    """
    Save the PyTorch model state to the given path.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Load metadata
seq_df = pd.read_csv(DATA_DIR / "sequences.csv", index_col=0)
if "sequence" not in seq_df.columns:
    seq_df = seq_df.rename(columns={seq_df.columns[0]: "sequence"})
pddt_df  = pd.read_csv(DATA_DIR / "pockets_mean_plddt.csv")
rmsd_df  = pd.read_csv(DATA_DIR / "pockets_rmsd.csv")
hack_df  = pd.read_csv(DATA_DIR / "hackathon-data(in).csv")

# Merge and label each pocket
metrics = (
    hack_df
      .merge(pddt_df, on=["uniprot_id","alphafold_pocket_selection"])  
      .merge(rmsd_df, on=["uniprot_id","alphafold_pocket_selection"])
      .merge(seq_df, on=['uniprot_id'])
)

def lent(x):
    try:
        return len(x)
    except TypeError as er:
        return 0

metrics['seq_len']=metrics['sequence'].apply(lent)

def trim_group(group_df):
    n = len(group_df)
    n_trim = n - (n % 16)  # largest multiple of 16 ≤ n
    return group_df.iloc[:n_trim]

# Apply to each group and concatenate

#metrics=metrics.drop('seq_len')

# fix suffixes: keep original pdb_pocket_selection from hack_df
if 'pdb_pocket_selection_x' in metrics.columns:
    metrics = metrics.rename(columns={'pdb_pocket_selection_x':'pdb_pocket_selection'})
elif 'pdb_pocket_selection' not in metrics.columns:
    raise KeyError('pdb_pocket_selection column missing after merge')
for col in ['pdb_pocket_selection_y']:
    if col in metrics.columns:
        metrics = metrics.drop(columns=[col])

metrics["is_cryptic"] = ((metrics["mean_plddt"] < 70.0) |
                          (metrics["pocket_rmsd"]  > 2.0)).astype(int)
metrics["alphafold_id"] = (metrics["alphafold_pocket_selection"]
                              .str.split(" and",n=1)
                              .str[0].str.strip())

# Fetch missing sequences from AlphaFold DB
missing = set(metrics["alphafold_id"]) - set(seq_df.index)
if missing:
    print(f"Fetching {len(missing)} missing sequences with 20 threads...")
    def fetch_seq(af_id):
        up = af_id.split('-')[1]
        url = f"https://alphafold.ebi.ac.uk/api/prediction/{up}"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code==200:
                data = r.json()
                seq = data[0].get("uniprotSequence")
                return af_id, seq
        except Exception:
            pass
        return af_id, None

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=20) as ex:
        for i, fut in enumerate(ex.map(fetch_seq, missing), start=1):
            af_id, seq = fut
            if seq:
                seq_df.loc[af_id] = seq
                print(f"[{i}/{len(missing)}] Fetched {af_id}")
            else:
                print(f"[{i}/{len(missing)}] Failed {af_id}")
    seq_df.to_csv(DATA_DIR / "sequences.csv")
    print("Updated sequences.csv")
    still = [x for x in missing if x not in seq_df.index]
    if still:
        print(f"Dropping {len(still)} pockets: {still}")
        metrics = metrics[~metrics["alphafold_id"].isin(still)]

# Split data
train_df, test_df = train_test_split(
    metrics, test_size=0.2,
    stratify=metrics["is_cryptic"],
    random_state=42
)
train_df = train_df.groupby('seq_len', group_keys=False).apply(trim_group).reset_index(drop=True)

train_df=train_df.sort_values(by="seq_len")


# Helper - parse PDB pocket indices
def parse_indices(sel:str):
    m = re.search(r"resi\s*([0-9+]+)", sel)
    if not m: return []
    return [int(x)-1 for x in m.group(1).split('+')]

# Main - load model, embed, dataset, train
if __name__ == "__main__":
    # Load ESM-2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = torch.hub.load(
        "facebookresearch/esm:main",
        "esm2_t33_650M_UR50D"
    )
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    # Embed full sequence -> [L, D]
    def embed_sequence(name, seq):
        max_len=1022; chunks=[]
        for i in range(0,len(seq),max_len):
            sub = seq[i:i+max_len]
            _,_,tokens = batch_converter([(name,sub)])
            tokens=tokens.to(device)
            with torch.no_grad():
                out = model(tokens, repr_layers=[model.num_layers])
            r = out["representations"][model.num_layers][0,1:-1]
            chunks.append(r.cpu())
        return torch.cat(chunks, dim=0)

    # Dataset for per-residue labels
    class PocketResidueDataset(Dataset):
        def __init__(self, df):
            self.df = df.reset_index(drop=True)
        def __len__(self): 
            return len(self.df)
        def __getitem__(self, i):
            row = self.df.loc[i]
            af  = row["alphafold_id"]
            seq = seq_df.loc[af, "sequence"]
            emb = embed_sequence(af, seq)
            L,D = emb.shape
            lbl = torch.zeros(L)
            for idx in parse_indices(row["pdb_pocket_selection"]):
                if 0 <= idx < L: lbl[idx]=1
            return emb, lbl

    train_ds = PocketResidueDataset(train_df)
    test_ds  = PocketResidueDataset(test_df)
    train_loader=DataLoader(train_ds,batch_size=16,shuffle=False)
    test_loader =DataLoader(test_ds,batch_size=1)

    class Transformer(nn.Module):
        def __init__(self,enc_dim=512,heads=8,hidden_ff_dim=2048,e_layers=1,dropout=0.2):
            super(Transformer, self).__init__()
        
            if hidden_ff_dim== None:
                hidden_ff_dim=4*enc_dim
            
            self.attention=Attention_Layer(Full_Attention(attention_dropout=dropout),enc_dim,heads,dropout)
            self.feedforward=FeedForward(enc_dim, hidden_ff_dim, dropout)
            #self.vocab_size=4**args.kmer
            #self.embedding = nn.Embedding(self.vocab_size, args.enc_dim)    
            self.encoder = nn.ModuleList([Encoder_layer(self.attention,self.feedforward,enc_dim,dropout) for _ in range(e_layers)])
            self.fc1_out = nn.Linear(enc_dim,1)

        def forward(self, x):
            #x = self.embedding(x)
            for layer in self.encoder:
                x = layer(x)
            x = self.fc1_out(x)
            return x.squeeze(-1)


    clf = Transformer(enc_dim=model.embed_dim,hidden_ff_dim=None,e_layers=1).to(device)
    crit=nn.BCEWithLogitsLoss()
    opt=torch.optim.Adam(clf.parameters(),lr=1e-4)

    # Train loop
    total_batches = len(train_loader)
    for ep in range(1, 6):
        clf.train()
        print(f"=== Training Epoch {ep}/5 ===")
        for batch_idx, (emb, lbl) in enumerate(train_loader, start=1):
            emb, lbl = emb.to(device), lbl.to(device)
            opt.zero_grad()
            #with autocast('cuda'):
            logits = clf(emb)
            loss = crit(logits, lbl)
            #scaler.scale(loss).backward()
            #scaler.step(opt)
            #scaler.update()

            if batch_idx % 10 == 0 or batch_idx == total_batches:
                print(f"[Epoch {ep}] Batch {batch_idx}/{total_batches}  Loss: {loss.item():.4f}")
            #del loss, logits
            #torch.cuda.empty_cache()
        # Save checkpoint after each epoch
        checkpoint_path = MODEL_DIR / f"clf_epoch{ep}.pth"
        save_model(clf, checkpoint_path)
        print(f"Epoch {ep} done and model saved")

    # Save final model
    final_path = MODEL_DIR / "clf_final.pth"
    save_model(clf, final_path)

    # Evaluate
    clf.eval()
    ys, ts = [], []
    with torch.no_grad():
        for emb, lbl in test_loader:
            emb = emb.to(device)
            logits = clf(emb)
            preds = torch.sigmoid(logits)[0]
            ys.append(preds.cpu().numpy())
            ts.append(lbl.numpy())
    y = np.concatenate(ys)
    t = np.concatenate(ts)
    print("ROC-AUC:", roc_auc_score(t, y))
    print("Acc@0.5:", accuracy_score(t, y > 0.5))
