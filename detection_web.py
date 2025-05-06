import os
import shutil
from time import time
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

from detection import detect   # your detection.py’s detect(...)

# determine project root (where this file lives)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# where uploaded known‑real and to‑check images go
REAL_DIR = os.path.join(BASE_DIR, "uploads", "known_real")
TEST_DIR = os.path.join(BASE_DIR, "uploads", "to_check")
# where Flask will serve static files (including our PCA plot)
STATIC_DIR = os.path.join(BASE_DIR, "static")

# make sure all directories exist
for d in (REAL_DIR, TEST_DIR, STATIC_DIR):
    os.makedirs(d, exist_ok=True)

# create Flask app, tell it where "static" is
app = Flask(__name__, static_folder=STATIC_DIR)

@app.route("/", methods=["GET", "POST"])
def index():
    results_table = []
    plot_url = None

    if request.method == "POST":
        # clear out any old uploads
        shutil.rmtree(REAL_DIR, ignore_errors=True)
        shutil.rmtree(TEST_DIR, ignore_errors=True)
        os.makedirs(REAL_DIR, exist_ok=True)
        os.makedirs(TEST_DIR, exist_ok=True)

        # save new known‑real files
        for f in request.files.getlist("real_signatures"):
            fname = secure_filename(f.filename)
            f.save(os.path.join(REAL_DIR, fname))

        # save new test files
        for f in request.files.getlist("test_signatures"):
            fname = secure_filename(f.filename)
            f.save(os.path.join(TEST_DIR, fname))

        # run detection; this will also save static/pca_plot.png
        det = detect(
            real_dir=REAL_DIR,
            test_dir=TEST_DIR,
            model_fp="signature_cnn.pth",
            compare_fp="compare_head.pth",
            cls_fp="cls_head.pth",
            embedding_dim=128,
            device="cpu"
        )

        # build results table: (filename, verdict, score)
        for path, metrics in det.items():
            fname   = os.path.basename(path)
            verdict = metrics[0]
            score   = metrics[-1]
            results_table.append((fname, verdict, f"{score:.3f}"))

        # cache‑bust the PCA plot URL
        plot_url = url_for("static", filename="pca_plot.png") + "?t=" + str(int(time()))

    return render_template("index.html",
                           results=results_table,
                           plot_url=plot_url)

if __name__ == "__main__":
    app.run(debug=True)
