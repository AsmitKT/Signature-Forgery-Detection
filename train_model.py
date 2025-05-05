# train_model.py

import os
import random
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from signature_embedding_cnn import SignatureEmbeddingCNN
from triplet_loss import TripletLoss
from ImagePreProcessing import process_signature

class SignatureTripletDataset(Dataset):
    """
    Builds (anchor, positive, negative) triplets from a folder structure:
      <root_dir>/
        ├─ 1/
        │    ├─ origi_xx.png
        │    ├─ forge_yy.png
        ├─ 2/
        │    ├─ origi_xx.png
        │    ├─ forge_yy.png
        ...
    Handles any number of originals/forgeries per subfolder:
      • If ≥2 originals and ≥1 forgeries: uses all ordered real pairs and all forgeries
      • If exactly 1 original and ≥1 forgeries: uses the same real as anchor/positive with augmentation
      • Skips if no forgeries or no originals
    """
    def __init__(self, root_dir: str, augment: bool = True):
        self.augment = augment
        self.triplets = []
        persons = [d for d in os.listdir(root_dir)
                   if os.path.isdir(os.path.join(root_dir, d))]
        for pid in persons:
            pdir = os.path.join(root_dir, pid)
            reals = [os.path.join(pdir, f)
                     for f in os.listdir(pdir)
                     if f.lower().startswith("origi")]
            forges = [os.path.join(pdir, f)
                      for f in os.listdir(pdir)
                      if f.lower().startswith("forge")]
            if not reals or not forges:
                continue
            if len(reals) >= 2:
                for a, p in itertools.permutations(reals, 2):
                    for n in forges:
                        self.triplets.append((a, p, n))
            else:
                a = p = reals[0]
                for n in forges:
                    self.triplets.append((a, p, n))

    def __len__(self):
        return len(self.triplets)

    def random_augment(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > 0.5:
            img = img.flip(-1)
        if random.random() > 0.5:
            img = img.flip(-2)
        if random.random() > 0.5:
            img = torch.rot90(img, k=2, dims=(-2, -1))
        return img

    def __getitem__(self, idx: int):
        a_path, p_path, n_path = self.triplets[idx]
        a_np = process_signature(a_path)
        p_np = process_signature(p_path)
        n_np = process_signature(n_path)

        a = torch.from_numpy(a_np).float().unsqueeze(0)
        p = torch.from_numpy(p_np).float().unsqueeze(0)
        n = torch.from_numpy(n_np).float().unsqueeze(0)

        if self.augment:
            a = self.random_augment(a)
            p = self.random_augment(p)
            n = self.random_augment(n)

        return a, p, n


def train_model(
    db_path:       str,
    embedding_dim: int   = 128,
    margin:        float = 1.5,
    epochs:        int   = 20,
    batch_size:    int   = 32,
    lr:            float = 1e-4,
    device:        str   = "cpu"
):
    # 1) Create or load model
    model_fp = "signature_cnn.pth"
    model = SignatureEmbeddingCNN(embedding_dim).to(device)
    if os.path.exists(model_fp):
        model.load_state_dict(torch.load(model_fp, map_location=device))

    # 2) Loss and optimizer
    loss_fn = TripletLoss(margin).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # 3) Report dataset info
    print(f"Training model on dataset: {db_path}")
    dataset = SignatureTripletDataset(db_path, augment=True)
    num_triplets = len(dataset)
    print(f"Found {num_triplets} triplets.")
    if num_triplets == 0:
        raise RuntimeError(f"No triplets found in {db_path}. Check your folder names and file prefixes.")

    # 4) List a few sample triplets
    print("Example triplets:")
    for a_path, p_path, n_path in dataset.triplets[:5]:
        print(f"  {os.path.basename(a_path)}, {os.path.basename(p_path)}, {os.path.basename(n_path)}")

    # 5) DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4
    )
    num_batches = len(loader)
    print(f"{num_batches} batches per epoch (batch_size={batch_size}).\n")

    # 6) Training loop with batch-hard negative mining and embedding normalization
    model.train()
    for epoch in range(1, epochs + 1):
        print(f"Starting epoch {epoch}/{epochs} on dataset folder: {os.path.basename(db_path)}")
        total_loss = 0.0
        for batch_idx, (A, P, N) in enumerate(loader, start=1):
            A, P, N = A.to(device), P.to(device), N.to(device)
            # forward pass + L2 normalize
            fA = F.normalize(model(A), p=2, dim=1)
            fP = F.normalize(model(P), p=2, dim=1)
            fN_all = F.normalize(model(N), p=2, dim=1)
            # batch-hard negative mining
            dists_AN = torch.cdist(fA, fN_all, p=2)
            hard_idx = torch.argmin(dists_AN, dim=1)
            hardN = fN_all[hard_idx]
            # compute loss
            loss = loss_fn(fA, fP, hardN)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 10 == 0 or batch_idx == num_batches:
                print(f"  Batch {batch_idx}/{num_batches} — loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch}/{epochs} — Avg Loss: {avg_loss:.4f}\n")

    # 7) Save weights
    torch.save(model.state_dict(), model_fp)
    print(f"Model weights saved to {model_fp}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Train SignatureEmbeddingCNN with Batch-Hard Triplet Loss"
    )
    parser.add_argument(
        "--database", type=str,
        default=r"C:\Coding\Python\Signature Forgery Detection\Database",
        help="Path to the root folder of your signature database"
    )
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--margin", type=float, default=1.5,
                        help="Triplet margin; larger values force harder separation")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="'cuda' or 'cpu'; auto-detected default"
    )
    args = parser.parse_args()

    train_model(
        db_path       = args.database,
        embedding_dim = args.embedding_dim,
        margin        = args.margin,
        epochs        = args.epochs,
        batch_size    = args.batch_size,
        lr            = args.lr,
        device        = args.device
    )
