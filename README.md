# Deepfake detection (paper re-implementation)

Python 3.10/3.11 recommended.

```
python -m venv .venv
.venv\Scripts\activate        (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
pip install torch torchvision   (use the exact command from pytorch.org for your GPU/CPU)

python deepfake.py "C:\path\to\dfdc-faces-of-the-train-sample" --smoke    # quick check (20 videos/class)
python deepfake.py "C:\path\to\dfdc-faces-of-the-train-sample"            # full run
```
The path is the folder that contains `train/`. Outputs go to `./work` (figures, ablation.csv, weights).
First run downloads dlib's 68-landmark model (needs internet). Preprocessing is cached after the first run.
Everything the paper leaves unspecified is marked `# ASSUMED` in the CFG block of deepfake.py.
