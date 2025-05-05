# Signature Forgery Detection

An end-to-end Python project for preprocessing handwritten signatures, training a deep CNN embedding model with triplet loss, and detecting forgeries using clustering-based methods.

## Table of Contents

* [Features](#features)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Usage](#usage)

  * [Data Preparation](#data-preparation)
  * [Training the Model](#training-the-model)
  * [Forgery Detection](#forgery-detection)
  * [DBSCAN Clustering & Visualization](#dbscan-clustering--visualization)
* [Dependencies](#dependencies)
* [Contributing](#contributing)
* [License](#license)

## Features

* **Image Preprocessing**: Adaptive local binarization, refinement, smoothing, cropping to fixed aspect ratio, resizing to 256×256 canvas.
* **Signature Embedding CNN**: Lightweight CNN producing D-dimensional embeddings from 1×256×256 inputs.
* **Triplet Loss Training**: Batch-hard mining with margin-based triplet loss for robust feature separation.
* **Forgery Detection**: Compare test signatures against known genuine set via DBSCAN clustering.
* **Clustering & Visualization**: 2D PCA scatter plots and distance histograms to diagnose embedding quality.

## Project Structure

```plaintext
├── Binary.py                # Local adaptive binarization & refinement
├── Crop.py                  # Crop to fixed aspect ratio without distortion
├── Resize.py                # Rotate, scale & center on 256×256 canvas
├── ImagePreProcessing.py    # End-to-end preprocessing pipeline
├── signature_embedding_cnn.py # Definition of the CNN embedding model
├── triplet_loss.py          # Triplet loss module
├── train_model.py           # Script to train embedding model
├── dbscan_clustering.py     # Cluster embeddings and plot results
├── detection.py             # Forgery detection using DBSCAN on embeddings
└── requirements.txt         # Project dependencies
```

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<YOUR_USERNAME>/signature-forgery-detection.git
   cd signature-forgery-detection
   ```
2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate      # Linux/macOS
   venv\\Scripts\\activate     # Windows
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation

Arrange your signature dataset as:

```
root_dir/
  ├─ 1/               # Person ID 1
  │    ├─ origi_*.png # Genuine signatures
  │    └─ forge_*.png # Forged signatures
  ├─ 2/               # Person ID 2
  │    ├─ origi_*.png
  │    └─ forge_*.png
  └─ ...
```

### Training the Model

Train the embedding CNN with triplet loss:

```bash
python train_model.py \
  --database path/to/root_dir \
  --embedding-dim 128 \
  --margin 1.5 \
  --epochs 20 \
  --batch-size 32 \
  --lr 1e-4 \
  --device cuda
```

Model weights are saved to `signature_cnn.pth` by default.

### Forgery Detection

Use known genuine signatures to classify new samples:

```bash
python detection.py \
  --real_dir path/to/KnownReal \
  --test_dir path/to/ToCheck \
  --model_fp signature_cnn.pth \
  --eps 1.2 \
  --min_samples 1 \
  --device cpu \
  --plot
```

Results are printed as `{image_path: "real"|"forged"}`.
Plots include PCA scatter and distance histograms when `--plot` is enabled.

### DBSCAN Clustering & Visualization

```python
from dbscan_clustering import cluster_and_plot
import numpy as np

# embeddings: numpy array of shape (N, D)
labels = cluster_and_plot(embeddings, eps=0.5, min_samples=5)
```

Produces a 2D PCA scatter colored by cluster labels.

## Dependencies

```text
python>=3.7
numpy
opencv-python
torch
torchvision
scikit-learn
matplotlib
```

See `requirements.txt` for exact versions.

## Contributing

Contributions, issues and feature requests are welcome!
Feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

*Author: AsmitKrT*
