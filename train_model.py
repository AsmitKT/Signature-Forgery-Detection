# train_model.py

import os
import random
import itertools
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from signature_embedding_cnn import SignatureEmbeddingCNN
from triplet_loss import TripletLoss
from ImagePreProcessing import process_signature
from sklearn.linear_model import LogisticRegression

class SignatureTripletDataset(Dataset):
    def __init__(self, root_dir, augment=True, max_triplets_per_folder=None):
        self.augment = augment
        self.triplets = []
        self.pids = []
        all_pids = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.pid2idx = {pid:i for i,pid in enumerate(sorted(all_pids))}
        for pid in sorted(all_pids):
            pdir = os.path.join(root_dir, pid)
            reals = [os.path.join(pdir,f) for f in os.listdir(pdir) if f.lower().startswith("origi")]
            forges = [os.path.join(pdir,f) for f in os.listdir(pdir) if f.lower().startswith("forge")]
            if not reals or not forges:
                continue
            folder = []
            for a,p in itertools.permutations(reals,2):
                for n in forges:
                    folder.append((a,p,n,pid))
            if max_triplets_per_folder and len(folder)>max_triplets_per_folder:
                folder = random.sample(folder, max_triplets_per_folder)
            for a,p,n,pid in folder:
                self.triplets.append((a,p,n))
                self.pids.append(self.pid2idx[pid])

    def __len__(self):
        return len(self.triplets)

    def random_augment(self, img):
        if random.random()>0.5: img=img.flip(-1)
        if random.random()>0.5: img=img.flip(-2)
        if random.random()>0.5: img=torch.rot90(img,2,(-2,-1))
        return img

    def __getitem__(self, idx):
        a,p,n = self.triplets[idx]
        pid    = self.pids[idx]
        A = torch.from_numpy(process_signature(a)).float().unsqueeze(0)
        P = torch.from_numpy(process_signature(p)).float().unsqueeze(0)
        N = torch.from_numpy(process_signature(n)).float().unsqueeze(0)
        if self.augment:
            A,P,N = self.random_augment(A), self.random_augment(P), self.random_augment(N)
        return A,P,N,torch.tensor(pid, dtype=torch.long)

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))
    def forward(self, feats, labels):
        return F.mse_loss(feats, self.centers[labels])

class SignatureSiameseModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.embed   = SignatureEmbeddingCNN(embedding_dim)
        self.compare = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # new classification head
        self.cls_head = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        z = self.embed(x)
        z = F.normalize(z, p=2, dim=1)
        return z

    def classify(self, z1, z2):
        return self.compare(torch.abs(z1 - z2)).squeeze(1)

    def classify_single(self, z):
        return self.cls_head(z).squeeze(1)

def train_model(db_path,
                embedding_dim=128,
                margin=2.0,
                epochs=20,
                batch_size=32,
                lr=1e-4,
                device="cpu",
                max_triplets_per_folder=None):
    device = torch.device(device)
    print(f"Loading dataset from {db_path}")
    ds = SignatureTripletDataset(db_path, augment=True, max_triplets_per_folder=max_triplets_per_folder)
    print(f" → {len(ds)} triplets, {len(ds.pid2idx)} writers")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
    print(f" → {len(loader)} batches/epoch of size {batch_size}\n")

    model = SignatureSiameseModel(embedding_dim).to(device)
    if os.path.exists("signature_cnn.pth"):
        model.embed.load_state_dict(torch.load("signature_cnn.pth", map_location=device))
    if os.path.exists("compare_head.pth"):
        model.compare.load_state_dict(torch.load("compare_head.pth", map_location=device))
    # no existing cls_head to load

    triplet_fn  = TripletLoss(margin).to(device)
    siamese_bce = nn.BCEWithLogitsLoss().to(device)
    cls_bce     = nn.BCEWithLogitsLoss().to(device)
    center_fn   = CenterLoss(len(ds.pid2idx), embedding_dim, device)
    optimizer   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    α, β, γ = 1.0, 0.05, 1.0
    model.train()
    for ep in range(1, epochs+1):
        print(f"=== Epoch {ep}/{epochs} ===")
        running = 0.0
        for i, (A,P,N,labels) in enumerate(loader, start=1):
            A,P,N,labels = A.to(device), P.to(device), N.to(device), labels.to(device)
            fA, fP = model(A), model(P)
            fN_all  = model(N)
            # hard negative
            dists = torch.cdist(fA, fN_all)
            hardN = fN_all[torch.argmin(dists,dim=1)]
            # triplet
            lt = triplet_fn(fA, fP, hardN)
            # siamese
            lp = siamese_bce(model.classify(fA,fP), torch.ones_like(labels, dtype=torch.float))
            ln = siamese_bce(model.classify(fA,hardN), torch.zeros_like(labels, dtype=torch.float))
            ls = lp + ln
            # center
            lc = center_fn(fA, labels)
            # classification head
            logP = model.classify_single(fP)
            logN = model.classify_single(hardN)
            lcp  = cls_bce(logP, torch.ones_like(logP))
            lcn  = cls_bce(logN, torch.zeros_like(logN))
            lcls = lcp + lcn
            # total
            loss = lt + α*ls + β*lc + γ*lcls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
            if i%10==0 or i==len(loader):
                print(f"Batch {i}/{len(loader)} — total:{loss.item():.4f} trip:{lt.item():.4f} siam:{ls.item():.4f} center:{lc.item():.4f} cls:{lcls.item():.4f}")

        print(f"Epoch {ep} avg loss: {running/len(loader):.4f}\n")

    # save
    torch.save(model.embed.state_dict(),    "signature_cnn.pth");    print("Saved embedding → signature_cnn.pth")
    torch.save(model.compare.state_dict(),  "compare_head.pth");    print("Saved siamese head → compare_head.pth")
    torch.save(model.cls_head.state_dict(), "cls_head.pth");        print("Saved cls head → cls_head.pth")
    torch.save(center_fn.centers.data,      "centers.pth");         print("Saved centers → centers.pth")

    # meta‑features & calibrator (unchanged) …
    # … same as before …

if __name__=="__main__":
    import argparse
    p = argparse.ArgumentParser(description="Train Signature Forgery Detection")
    p.add_argument("--database",                type=str,   default="./Database")
    p.add_argument("--embedding-dim",           type=int,   default=128)
    p.add_argument("--margin",                  type=float, default=2.0)
    p.add_argument("--epochs",                  type=int,   default=30)
    p.add_argument("--batch-size",              type=int,   default=32)
    p.add_argument("--lr",                      type=float, default=1e-4)
    p.add_argument("--device",                  type=str,   default="cpu")
    p.add_argument("--max-triplets-per-folder", type=int,   default=10)
    args = p.parse_args()
    train_model(
      args.database, args.embedding_dim, args.margin,
      args.epochs, args.batch_size, args.lr,
      args.device, args.max_triplets_per_folder
    )
