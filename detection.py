# detection.py
import os
import glob
import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from train_model import SignatureSiameseModel
from ImagePreProcessing import process_signature

def load_model(model_fp, compare_fp, cls_fp, embedding_dim, device):
    dev   = torch.device(device)
    model = SignatureSiameseModel(embedding_dim).to(dev)
    model.embed.load_state_dict(torch.load(model_fp,   map_location=dev, weights_only=True), strict=True)
    model.compare.load_state_dict(torch.load(compare_fp, map_location=dev, weights_only=True), strict=True)
    model.cls_head.load_state_dict(torch.load(cls_fp,    map_location=dev, weights_only=True), strict=True)
    model.eval()
    return model, dev

def detect(real_dir, test_dir,
           model_fp="signature_cnn.pth",
           compare_fp="compare_head.pth",
           cls_fp="cls_head.pth",
           embedding_dim=128,
           device="cpu"):

    model, dev = load_model(model_fp, compare_fp, cls_fp, embedding_dim, device)

    # ── embed support signatures ──────────────────────────────────────────────────
    support_paths = sorted(glob.glob(os.path.join(real_dir, "*.*")))
    support = []
    with torch.no_grad():
        for p in support_paths:
            arr = process_signature(p)
            t   = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float().to(dev)
            support.append(model(t))
    support = torch.cat(support, dim=0)  # [K, D]
    print(f"Found {len(support_paths)} support signatures → embeddings shape: {support.shape}\n")

    # ── compute support’s own 3 features ──────────────────────────────────────────
    self_means, self_stds, self_cls = [], [], []
    with torch.no_grad():
        for z in support:
            diffs  = torch.abs(z.unsqueeze(0) - support)
            logits = model.compare(diffs).squeeze(1)
            sims   = torch.sigmoid(logits).cpu().numpy()
            self_means.append(sims.mean())
            self_stds.append(sims.std())
            self_cls.append(
                torch.sigmoid(model.cls_head(z.unsqueeze(0)).squeeze(1)).item()
            )

    # ── compute 4th feature: support‐centroid & its threshold ────────────────────
    centroid = support.mean(dim=0, keepdim=True)            # [1, D]
    dists    = torch.norm(support - centroid, dim=1).cpu().numpy()
    # e.g. allow 90th‐percentile distance
    dist_thr = np.percentile(dists, 90)
    print(f"Centroid‐distance threshold (90th‑pct): {dist_thr:.3f}\n")

    # ── percentile thresholds for the other 3 ───────────────────────────────────
    std_thr  = np.percentile(self_stds,  10)   # e.g. 10th‑pct
    cls_thr  = np.percentile(self_cls,   100)   # e.g. 30th‑pct
    mean_thr = np.percentile(self_means, 30)   # e.g. 20th‑pct

    print("Percentile thresholds:")
    print(f"  std  ≥ 10th‑pct → {std_thr:.3f}")
    print(f"  cls_p≥ 100th‑pct → {cls_thr:.3f}")
    print(f"  mean ≥ 30th‑pct → {mean_thr:.3f}\n")

    # ── classify test signatures ───────────────────────────────────────────────────
    test_paths = sorted(glob.glob(os.path.join(test_dir, "*.*")))
    print(f"Found {len(test_paths)} test signatures\nClassifying:")
    results, test_pred = [], []
    with torch.no_grad():
        for p in test_paths:
            arr   = process_signature(p)
            t     = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float().to(dev)
            zq    = model(t)

            # Siamese mean/std
            diffs  = torch.abs(zq - support)
            logits = model.compare(diffs).squeeze(1)
            sims   = torch.sigmoid(logits).cpu().numpy()
            m, s   = sims.mean(), sims.std()

            # classification‐head
            cls_p  = torch.sigmoid(model.cls_head(zq).squeeze(1)).item()

            # distance‐to‑centroid
            d_q    = torch.norm(zq - centroid).item()

            # ── cascade with 4th feature fallback ───────────────────
            if d_q > dist_thr:
                verdict = "forge"
            elif (s >= std_thr) and (len(support_paths)>15):
                verdict = "real"
            elif (m < mean_thr) and (cls_p < cls_thr):
                verdict = "forged"
            else:
                verdict = "real"
            # ──────────────────────────────────────────────────────────

            print(f"  {os.path.basename(p)} → m={m:.3f}, s={s:.3f}, cls={cls_p:.3f}, d={d_q:.3f} → {verdict}")
            results.append((p, verdict, m, s, cls_p, d_q))
            test_pred.append(verdict=="real")

    print("\nDetection complete.\n")

    # ── PCA visualization ──────────────────────────────────────────────────────────
    sup_embs  = support.cpu().numpy()
    test_embs = []
    with torch.no_grad():
        for p, *_ in results:
            arr = process_signature(p)
            t   = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float().to(dev)
            test_embs.append(model(t).cpu().numpy().reshape(-1))

    X   = np.vstack((sup_embs, test_embs))
    X2  = PCA(n_components=2).fit_transform(X)
    sup2, tst2 = X2[:len(sup_embs)], X2[len(sup_embs):]

    plt.figure()
    plt.scatter(sup2[:,0], sup2[:,1], c='green', marker='o', label='support')
    for (x,y), is_real in zip(tst2, test_pred):
        plt.scatter(x, y, c='green' if is_real else 'red', marker='x')
    plt.title("PCA of Signature Embeddings")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend()
    plt.tight_layout()
    plt.show()

    return {p:(v,m,s,cls_p,d_q) for p,v,m,s,cls_p,d_q in results}


if __name__=="__main__":
    import argparse, pprint
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir",      default="./KnownReal")
    p.add_argument("--test_dir",      default="./ToCheck")
    p.add_argument("--model_fp",      default="signature_cnn.pth")
    p.add_argument("--compare_fp",    default="compare_head.pth")
    p.add_argument("--cls_fp",        default="cls_head.pth")
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--device",        default="cpu")
    args = p.parse_args()

    res = detect(
      args.real_dir, args.test_dir,
      args.model_fp, args.compare_fp, args.cls_fp,
      args.embedding_dim, args.device
    )
    #pprint.pprint(res)
