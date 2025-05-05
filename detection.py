import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

from signature_embedding_cnn import SignatureEmbeddingCNN
from ImagePreProcessing import process_signature

def detect_signatures(
    real_dir: str,
    test_dir: str,
    model_fp: str = "signature_cnn.pth",
    eps: float = 1.2,
    min_samples: int = 1,
    device: str = "cpu",
    plot: bool = False
) -> dict:
    """
    real_dir:   folder of known genuine signatures
    test_dir:   folder of signatures to classify
    plot:       if True, show embedding scatter and distance histograms
    returns:    dict { test_image_path: "real" or "forged" }
    """
    # 1) Load model (use weights_only=True to suppress pickle warning)
    model = SignatureEmbeddingCNN().to(device)
    state = torch.load(model_fp, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # 2) Gather image paths
    real_paths = [os.path.join(real_dir, f)
                  for f in os.listdir(real_dir)
                  if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    test_paths = [os.path.join(test_dir, f)
                  for f in os.listdir(test_dir)
                  if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    # 3) Helper: Embed a list of paths
    def embed_list(paths):
        embs = []
        with torch.no_grad():
            for p in paths:
                np_img = process_signature(p)
                if not isinstance(np_img, np.ndarray):
                    raise TypeError(f"process_signature should return a NumPy array, got {type(np_img)}")
                tensor_img = torch.tensor(np_img, dtype=torch.float32)
                if tensor_img.ndim == 2:
                    tensor_img = tensor_img.unsqueeze(0)
                tensor_img = tensor_img.unsqueeze(0).to(device)
                embedding = model(tensor_img).cpu().numpy().reshape(-1)
                embs.append(embedding)
        return np.vstack(embs)

    # 4) Compute embeddings
    real_embs = embed_list(real_paths)
    test_embs = embed_list(test_paths)

    # 5) L2-normalize embeddings for more stable distance metrics
    real_embs = normalize(real_embs, axis=1)
    test_embs = normalize(test_embs, axis=1)
    all_embs = np.vstack((real_embs, test_embs))

    # 6) Clustering with adjusted eps and min_samples
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(all_embs)
    labels = clustering.labels_
    real_labels = set(labels[:len(real_embs)])
    test_labels = labels[len(real_embs):]

    # 7) Interpret results
    results = {}
    for p, lbl in zip(test_paths, test_labels):
        if lbl in real_labels and lbl != -1:
            results[p] = "real"
        else:
            results[p] = "forged"

    # 8) Optional plotting for diagnosis
    if plot:
        # 8a) PCA scatter of embeddings
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(all_embs)
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced[:len(real_embs), 0],
                    reduced[:len(real_embs), 1],
                    label='Genuine', alpha=0.6, edgecolors='k')
        plt.scatter(reduced[len(real_embs):, 0],
                    reduced[len(real_embs):, 1],
                    label='To Check', marker='x')
        for idx, (x, y) in enumerate(reduced):
            plt.text(x, y, str(labels[idx]), fontsize=8)
        plt.legend()
        plt.title('PCA of Signature Embeddings with DBSCAN Labels')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.tight_layout()
        plt.show()

        # 8b) Histogram of min-distance to genuine
        dists = np.linalg.norm(test_embs[:, None, :] - real_embs[None, :, :], axis=2)
        min_dists = dists.min(axis=1)
        plt.figure(figsize=(6, 4))
        plt.hist(min_dists, bins=20)
        plt.title('Min L2-distance from Genuine Signatures')
        plt.xlabel('Distance')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()

    return results


if __name__ == "__main__":
    import pprint
    reals = r"C:\Coding\Python\Signature Forgery Detection\KnownReal"
    tests = r"C:\Coding\Python\Signature Forgery Detection\ToCheck"
    # set plot=True to visualize embeddings and distances
    res = detect_signatures(reals, tests, device="cpu", plot=True)
    pprint.pprint(res)
