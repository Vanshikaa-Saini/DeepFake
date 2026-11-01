import sys, os, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as _plt
_fig_n = [0]
def show():
    _fig_n[0] += 1; os.makedirs("work", exist_ok=True)
    _plt.savefig(f"work/figure_{_fig_n[0]:02d}.png", dpi=150, bbox_inches="tight"); _plt.close("all")

import os, re, json, bz2, math, random, urllib.request
from pathlib import Path
import numpy as np, pandas as pd, cv2
import matplotlib.pyplot as plt, seaborn as sns
from tqdm.auto import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

SEED = 42
def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print("device:", DEVICE, "| torch", torch.__version__)

CFG = dict(
    # ---------- data ----------
    DATA_ROOT   = (sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else None),
    WORK_DIR    = "./work_all",
    N_FRAMES    = 60,        # PAPER: fixed 60 frames sampled per video
    IMG_SIZE    = 128,       # PAPER: 128x128 input to VGG16
    BALANCE     = False,     # DEVIATION from paper: use ALL videos, handle imbalance with a class-weighted loss
    MAX_PER_CLASS = None,    # e.g. 30 for a quick smoke test; None = use all
    TEST_FRAC   = 0.20,      # PAPER: 80:20 train/test split
    VAL_FRAC    = 0.10,      # of the training part, used for early stopping / validation curves  # ASSUMED
    # ---------- model ----------
    LSTM_UNITS  = 128,       # ASSUMED (paper does not state)
    TF_HEADS    = 4,         # ASSUMED
    TF_LAYERS   = 1,         # ASSUMED
    TF_FF_DIM   = 128,       # ASSUMED (paper: key_dim = ff_dim)
    DROPOUT     = 0.5,       # PAPER: dropout 0.5 after each FC layer
    FINETUNE_VGG= False,     # ASSUMED: paper doesn't say; frozen ImageNet VGG16 is far cheaper
    DDM_MODE    = "eq17",    # "eq17": Z = mean_t |S_{t+1}-S_t| (Eq.17)  |  "concat": Eq.13-14 concat(z_t, |z_t-z_{t-1}|) then mean-pool
    # ---------- training ----------
    LR          = 1e-3,      # PAPER: initial learning rate 0.001 (optimizer assumed Adam)
    BATCH_SIZE  = 32,        # PAPER: DFDC best = 32 ; FF++/UADFV = 64
    EPOCHS      = 12,        # PAPER: DFDC best = 12 ; FF++/UADFV = 30
    PATIENCE    = 5,         # PAPER: early stopping on val loss (patience ASSUMED)
    AUGMENT     = True,      # PAPER: random rotation, flipping, brightness
)
if "--smoke" in sys.argv: CFG["MAX_PER_CLASS"] = 20
CFG.update(json.loads(os.environ.get("DF_OVERRIDES", "{}")))
os.makedirs(CFG["WORK_DIR"], exist_ok=True)

def find_data_root():
    for c in [CFG["DATA_ROOT"], "/kaggle/input/dfdc-faces-of-the-train-sample",
              "/content/dfdc-faces-of-the-train-sample", "/content", ".", "./dfdc-faces-of-the-train-sample"]:
        if c and os.path.isdir(os.path.join(c, "train")): return c
        if c and os.path.basename(os.path.normpath(c)) == "train" and os.path.isdir(c): return os.path.dirname(os.path.abspath(c))
    raise FileNotFoundError("Set CFG['DATA_ROOT'] to the folder that contains `train/`.")
DATA_ROOT = find_data_root(); TRAIN_DIR = os.path.join(DATA_ROOT, "train")
print("DATA_ROOT:", DATA_ROOT)
print("entries in train/:", len(os.listdir(TRAIN_DIR)), sorted(os.listdir(TRAIN_DIR))[:6])
print("next to train/:", [f for f in os.listdir(DATA_ROOT) if f != "train"])

IMG_EXT = (".jpg", ".jpeg", ".png")
def natural_key(p): return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(p))]

# Layout supported: train/real/... and train/fake/... (class = parent folder name).
# Inside a class folder, frames are either in one sub-folder per video, or flat files like <video>_<frame>.png
videos, labels = {}, {}
def vid_key(p):                                   # frame file -> video id (strip trailing _<digits>)
    return re.sub(r"(?:[_\-]\d+)+$", "", p.stem)
for cls_dir in sorted(p for p in Path(TRAIN_DIR).iterdir() if p.is_dir()):
    cname = cls_dir.name.lower()
    if "fake" in cname: y = "FAKE"
    elif "real" in cname: y = "REAL"
    else: continue
    # per-video sub-folders
    for d in sorted(p for p in cls_dir.rglob("*") if p.is_dir()):
        imgs = sorted([p for p in d.iterdir() if p.suffix.lower() in IMG_EXT], key=natural_key)
        if imgs: videos[f"{y}_{d.name}"] = imgs; labels[f"{y}_{d.name}"] = y
    # flat files directly inside the class folder
    for p in sorted([p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXT], key=natural_key):
        k = f"{y}_{vid_key(p)}"; videos.setdefault(k, []).append(p); labels[k] = y
print("videos with frames:", len(videos), "| label counts:", pd.Series(labels).value_counts().to_dict())
if not videos:
    print("DIAGNOSTIC - folder layout under", DATA_ROOT)
    for dp, dn, fn in list(os.walk(DATA_ROOT))[:25]: print("  ", os.path.relpath(dp, DATA_ROOT), "| subdirs:", len(dn), "| files:", len(fn), fn[:3])
    raise SystemExit("No frames found under train/real or train/fake. Paste the layout above.")
_fc = pd.Series({k: len(v) for k, v in videos.items()}); print("frames per video:", _fc.describe()[["min", "mean", "max"]].round(1).to_dict())
print("example video ids:", list(videos)[:3])

df = pd.DataFrame([dict(video=v, label=int(labels[v] in ("FAKE", "1")), n_frames=len(p)) for v, p in videos.items() if v in labels])
print(df.label.map({0: "REAL", 1: "FAKE"}).value_counts()); print(df.n_frames.describe())

if CFG["BALANCE"]:
    n = df.label.value_counts().min()
    if CFG["MAX_PER_CLASS"]: n = min(n, CFG["MAX_PER_CLASS"])
    df = pd.concat([g.sample(n, random_state=SEED) for _, g in df.groupby("label")]).reset_index(drop=True)
elif CFG["MAX_PER_CLASS"]:
    df = pd.concat([g.sample(min(len(g), CFG["MAX_PER_CLASS"]), random_state=SEED) for _, g in df.groupby("label")]).reset_index(drop=True)
print("videos used:", len(df), df.label.value_counts().to_dict())

trv, te = train_test_split(df, test_size=CFG["TEST_FRAC"], stratify=df.label, random_state=SEED)
tr, va = train_test_split(trv, test_size=CFG["VAL_FRAC"], stratify=trv.label, random_state=SEED)
POS_W = torch.tensor([(tr.label == 0).sum() / max((tr.label == 1).sum(), 1)], dtype=torch.float32)   # weight on FAKE = n_real/n_fake
print(f"class weight for FAKE samples (pos_weight): {POS_W.item():.3f}")
for nme, d in [("train", tr), ("val", va), ("test", te)]: print(f"{nme:5s} {len(d):4d} videos  fake-ratio {d.label.mean():.2f}")

import dlib
LM_PATH = os.path.join(CFG["WORK_DIR"], "shape_predictor_68_face_landmarks.dat")
if not os.path.exists(LM_PATH) and os.path.exists("./work/shape_predictor_68_face_landmarks.dat"):
    LM_PATH = "./work/shape_predictor_68_face_landmarks.dat"
if not os.path.exists(LM_PATH):
    print("downloading dlib 68-landmark model ...")
    urllib.request.urlretrieve("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2", LM_PATH + ".bz2")
    with bz2.open(LM_PATH + ".bz2") as fi, open(LM_PATH, "wb") as fo: fo.write(fi.read())
predictor = dlib.shape_predictor(LM_PATH)
detector  = dlib.get_frontal_face_detector()

# ---------- frame enhancement (paper Eq.1-4 + NLM + HSV boost) ----------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))        # clip/tile ASSUMED (paper gives none)
def enhance(bgr):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    ycc[:, :, 0] = clahe.apply(ycc[:, :, 0])                         # CLAHE on brightness channel only
    out = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
    out = cv2.fastNlMeansDenoisingColored(out, None, 5, 5, 7, 21)    # Non-Local Means (strengths ASSUMED)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255)               # raise saturation (factor ASSUMED)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.10, 0, 255)               # raise value/brightness
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# ---------- landmarks + geometric features ----------
def landmarks68(bgr):
    h, w = bgr.shape[:2]; gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    rect = max(rects, key=lambda r: r.area()) if len(rects) else dlib.rectangle(0, 0, w - 1, h - 1)   # crops are already faces
    s = predictor(gray, rect)
    return np.array([[s.part(i).x, s.part(i).y] for i in range(68)], dtype=np.float32)

def dist(P, a, b): return float(np.linalg.norm(P[a] - P[b]))
def angle(P, a, b, c):          # angle at vertex b (Eq. 7)
    v1, v2 = P[a] - P[b], P[c] - P[b]
    return float(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8), -1, 1)))

def geometric_features(P):
    # Distances / angles named in the paper: eye-eye, nose-mouth, eye widths, mouth width & height,
    # nose-to-chin (vertical), cheek-to-cheek (horizontal), eye-nose-eye angle, mouth-corner angles.
    # The exact full list is NOT given in the paper -> this list is ASSUMED/extended in the same spirit.
    reye, leye = P[36:42].mean(0), P[42:48].mean(0)
    face_w = dist(P, 1, 15) + 1e-6                                   # scale reference -> size invariance
    D = [np.linalg.norm(reye - leye),                                # distance between eyes
         dist(P, 36, 45),                                            # outer eye corners
         dist(P, 33, 51),                                            # nose - upper lip
         dist(P, 36, 39), dist(P, 42, 45),                           # width of each eye
         (dist(P, 37, 41) + dist(P, 38, 40)) / 2, (dist(P, 43, 47) + dist(P, 44, 46)) / 2,   # eye heights
         dist(P, 48, 54), dist(P, 51, 57), dist(P, 62, 66),          # mouth width / height / inner height
         dist(P, 33, 8), dist(P, 27, 8),                             # nose-to-chin / bridge-to-chin (face length)
         dist(P, 1, 15), dist(P, 0, 16),                             # cheek-to-cheek
         dist(P, 19, 37), dist(P, 24, 44),                           # eyebrow-eye
         dist(P, 31, 35), dist(P, 30, 33),                           # nose width / height
         dist(P, 4, 12), dist(P, 5, 11)]                             # jaw width
    D = [d / face_w for d in D]
    A = [angle(P, 36, 30, 45),                                       # right eye - nose tip - left eye
         angle(P, 48, 51, 54), angle(P, 48, 57, 54),                 # mouth corners vs upper/lower lip
         angle(P, 4, 8, 12), angle(P, 36, 33, 45), angle(P, 48, 33, 54),
         angle(P, 17, 27, 26), angle(P, 0, 30, 16)]
    return np.array(D + A, dtype=np.float32)
N_GEO = len(geometric_features(np.random.rand(68, 2).astype(np.float32) * 100)); print("geometric features per frame:", N_GEO)

def sample_paths(paths, T):
    return [paths[i] for i in np.linspace(0, len(paths) - 1, T).round().astype(int)]   # evenly spaced, repeats if video is shorter

def prep_video(vid):
    S = CFG["IMG_SIZE"]; fr, geo = [], []
    for p in sample_paths(videos[vid], CFG["N_FRAMES"]):
        bgr = cv2.imread(str(p))
        if bgr is None: bgr = np.zeros((S, S, 3), np.uint8)
        e = enhance(cv2.resize(bgr, (S, S)) if min(bgr.shape[:2]) < 64 else bgr)
        e = cv2.resize(e, (S, S), interpolation=cv2.INTER_AREA)
        geo.append(geometric_features(landmarks68(e)))
        fr.append(cv2.cvtColor(e, cv2.COLOR_BGR2RGB))
    return np.stack(fr), np.stack(geo)

PREP = os.path.join(CFG["WORK_DIR"], f"prep_T{CFG['N_FRAMES']}_S{CFG['IMG_SIZE']}_{len(df)}.npz")
if os.path.exists(PREP):
    z = np.load(PREP, allow_pickle=True); names = list(z["names"])
    _fa, _ga = z["frames"], z["geo"]; FRAMES = {n: _fa[i] for i, n in enumerate(names)}; GEO = {n: _ga[i] for i, n in enumerate(names)}
    print("loaded cache:", len(names), "videos")
else:
    FRAMES, GEO = {}, {}
    for v in tqdm(df.video, desc="preprocessing"): FRAMES[v], GEO[v] = prep_video(v)
    names = list(FRAMES)
    np.savez(PREP, names=np.array(names), frames=np.stack([FRAMES[n] for n in names]), geo=np.stack([GEO[n] for n in names]))
    print("saved ->", PREP)

# z-score the geometric features with TRAIN statistics only
g_all = np.concatenate([GEO[v] for v in tr.video]); G_MEAN, G_STD = g_all.mean(0), g_all.std(0) + 1e-6

v0 = df.video.iloc[0]
raw = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in sample_paths(videos[v0], 4)]
fig, ax = plt.subplots(2, 4, figsize=(12, 6))
for i in range(4):
    ax[0, i].imshow(raw[i]); ax[0, i].set_title("original" if i == 0 else ""); ax[0, i].axis("off")
    ax[1, i].imshow(FRAMES[v0][i * (CFG["N_FRAMES"] // 4)]); ax[1, i].set_title("enhanced 128x128" if i == 0 else ""); ax[1, i].axis("off")
plt.tight_layout(); show()
P = landmarks68(cv2.cvtColor(FRAMES[v0][0], cv2.COLOR_RGB2BGR)); im = FRAMES[v0][0].copy()
for x, y in P.astype(int): cv2.circle(im, (x, y), 1, (0, 255, 0), -1)
plt.figure(figsize=(3, 3)); plt.imshow(im); plt.axis("off"); plt.title("68 landmarks"); show()

class PosEnc(nn.Module):
    def __init__(self, d, maxlen=512):
        super().__init__()
        pe = torch.zeros(maxlen, d); pos = torch.arange(maxlen).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div[: d // 2])
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, : x.size(1)]

class TransformerBlock(nn.Module):
    """Paper Algorithm 1: MHA -> dropout -> residual+LN -> FFN(ReLU) -> dropout -> residual+LN."""
    def __init__(self, d, heads, ff, p):
        super().__init__()
        self.att = nn.MultiheadAttention(d, heads, dropout=0.0, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Linear(ff, d))
        self.n1, self.n2, self.dp = nn.LayerNorm(d, eps=1e-6), nn.LayerNorm(d, eps=1e-6), nn.Dropout(p)
    def forward(self, x):
        x = self.n1(x + self.dp(self.att(x, x, x, need_weights=False)[0]))
        return self.n2(x + self.dp(self.ffn(x)))

def ddm(z, mode="eq17"):
    d = (z[:, 1:] - z[:, :-1]).abs()                       # |S_{t+1}-S_t|, (B,T-1,D)
    if mode == "eq17": return d.mean(1)                    # Eq.17
    return torch.cat([z[:, 1:], d], -1).mean(1)            # Eq.13-14: concat(z_t, d_t), then mean-pool

class DeepfakeNet(nn.Module):
    def __init__(self, use_landmarks=True, use_transformer=True, use_ddm=True):
        super().__init__()
        c = CFG; self.use_lm, self.use_tf, self.use_ddm = use_landmarks, use_transformer, use_ddm
        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        self.cnn = vgg.features                                   # FC layers removed (paper)
        for p in self.cnn.parameters(): p.requires_grad = c["FINETUNE_VGG"]
        in_dim = 512 + (N_GEO if use_landmarks else 0)
        self.lstm = nn.LSTM(in_dim, c["LSTM_UNITS"], batch_first=True)
        K = c["LSTM_UNITS"]; self.pos = PosEnc(K)
        self.tf = nn.Sequential(*[TransformerBlock(K, c["TF_HEADS"], c["TF_FF_DIM"], c["DROPOUT"]) for _ in range(c["TF_LAYERS"])]) if use_transformer else nn.Identity()
        out = K * (2 if (use_ddm and c["DDM_MODE"] == "concat") else 1)
        self.head = nn.Sequential(nn.Linear(out, 64), nn.ReLU(), nn.Dropout(c["DROPOUT"]), nn.Linear(64, 1))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, frames, geo):             # frames: (B,T,3,H,W) in [0,1] ; geo: (B,T,N_GEO)
        B, T = frames.shape[:2]
        x = ((frames.flatten(0, 1) - self.mean) / self.std)
        with torch.set_grad_enabled(torch.is_grad_enabled() and CFG["FINETUNE_VGG"]):
            f = torch.cat([F.adaptive_avg_pool2d(self.cnn(c), 1).flatten(1) for c in x.split(64)]).view(B, T, -1)     # (B,T,512), chunked
        if self.use_lm: f = torch.cat([f, geo], -1)
        h, _ = self.lstm(f)                                                         # H_LSTM (B,T,K)
        z = self.tf(self.pos(h))                                                    # X = H + PE -> Transformer
        v = ddm(z, CFG["DDM_MODE"]) if self.use_ddm else z.mean(1)
        return self.head(v).squeeze(1)                                              # logit; sigmoid applied in loss / at inference

m = DeepfakeNet().to(DEVICE)
print("trainable params: %.2fM" % (sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6))
with torch.no_grad():
    print("output:", m(torch.rand(2, CFG["N_FRAMES"], 3, CFG["IMG_SIZE"], CFG["IMG_SIZE"], device=DEVICE),
                       torch.randn(2, CFG["N_FRAMES"], N_GEO, device=DEVICE)).shape)
del m

def make_batch(vids, train):
    fr = torch.from_numpy(np.stack([FRAMES[v] for v in vids])).to(DEVICE).permute(0, 1, 4, 2, 3).float() / 255.   # (B,T,3,H,W)
    geo = torch.from_numpy(np.stack([(GEO[v] - G_MEAN) / G_STD for v in vids])).float().to(DEVICE)
    if train and CFG["AUGMENT"]:
        B, T = fr.shape[:2]
        for b in range(B):
            clip = fr[b]
            if random.random() < 0.5: clip = TF.hflip(clip)
            clip = TF.rotate(clip, random.uniform(-10, 10))
            clip = TF.adjust_brightness(clip, random.uniform(0.8, 1.2))
            fr[b] = clip.clamp(0, 1)
    return fr, geo

def batches(d, train, bs=None):
    bs = bs or CFG["BATCH_SIZE"]; idx = np.random.permutation(len(d)) if train else np.arange(len(d))
    for i in range(0, len(d), bs):
        sub = d.iloc[idx[i:i + bs]]
        fr, geo = make_batch(sub.video.tolist(), train)
        yield fr, geo, torch.tensor(sub.label.values, dtype=torch.float32, device=DEVICE)

@torch.no_grad()
def run_eval(model, d, bs=8):
    model.eval(); P, Y, L = [], [], 0.
    for fr, geo, y in batches(d, False, bs):
        lo = model(fr, geo); L += F.binary_cross_entropy_with_logits(lo, y, reduction="sum", pos_weight=POS_W.to(DEVICE)).item()
        P.append(torch.sigmoid(lo).cpu()); Y.append(y.cpu())
    return torch.cat(P).numpy(), torch.cat(Y).numpy(), L / len(d)

def train_model(name, **flags):
    seed_all(); model = DeepfakeNet(**flags).to(DEVICE)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=CFG["LR"])
    H = dict(tr_loss=[], va_loss=[], tr_acc=[], va_acc=[]); best, bad = 1e9, 0
    ck = os.path.join(CFG["WORK_DIR"], f"{name}.pt")
    for ep in range(1, CFG["EPOCHS"] + 1):
        model.train(); L = C = 0
        for fr, geo, y in batches(tr, True):
            opt.zero_grad(); lo = model(fr, geo); loss = F.binary_cross_entropy_with_logits(lo, y, pos_weight=POS_W.to(DEVICE)); loss.backward(); opt.step()
            L += loss.item() * len(y); C += ((lo > 0).float() == y).sum().item()
        p, y, vl = run_eval(model, va)
        H["tr_loss"].append(L / len(tr)); H["tr_acc"].append(C / len(tr)); H["va_loss"].append(vl); H["va_acc"].append(((p >= .5) == y).mean())
        print(f"[{name}] epoch {ep:02d}  loss {L/len(tr):.4f} acc {C/len(tr):.3f} | val loss {vl:.4f} acc {H['va_acc'][-1]:.3f} auc {(roc_auc_score(y, p) if len(set(y)) > 1 else float('nan')):.3f}")
        if vl < best: best, bad = vl, 0; torch.save(model.state_dict(), ck)
        else:
            bad += 1
            if bad >= CFG["PATIENCE"]: print("early stopping"); break
    model.load_state_dict(torch.load(ck, map_location=DEVICE)); return model, H

def plot_hist(H, name):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(H["tr_acc"], label="train"); ax[0].plot(H["va_acc"], label="val"); ax[0].set_title(f"{name}: accuracy"); ax[0].legend()
    ax[1].plot(H["tr_loss"], label="train"); ax[1].plot(H["va_loss"], label="val"); ax[1].set_title(f"{name}: loss"); ax[1].legend()
    for a in ax: a.set_xlabel("epoch")
    plt.tight_layout(); plt.savefig(os.path.join(CFG["WORK_DIR"], f"{name}_curves.png"), dpi=150); show()

def _metrics(y, p, thr):
    yh = (p > thr).astype(int)
    return dict(acc=accuracy_score(y, yh), bal_acc=balanced_accuracy_score(y, yh),
                precision=precision_score(y, yh, zero_division=0), recall=recall_score(y, yh, zero_division=0),
                f1=f1_score(y, yh, zero_division=0), auc=roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")), yh

def test_report(model, name):
    # threshold 0.5 (paper) + a threshold tuned on the VALIDATION set only (max balanced accuracy), applied to test
    pv, yv, _ = run_eval(model, va)
    cands = np.unique(np.concatenate([pv, [0.5]]))
    thr = float(max(cands, key=lambda t: (balanced_accuracy_score(yv, (pv > t).astype(int)), -abs(t - 0.5)))) if len(set(yv)) > 1 else 0.5
    p, y, _ = run_eval(model, te)
    m05, yh = _metrics(y, p, 0.5); mt, yht = _metrics(y, p, thr)
    r = dict(model=name, **{k: v for k, v in m05.items()}, val_tuned_thr=thr, **{f"{k}@thr": v for k, v in mt.items() if k != "auc"})
    return r, y, yh, p

full, H = train_model("full", use_landmarks=True, use_transformer=True, use_ddm=True)
plot_hist(H, "full")
res, y, yh, p_full = test_report(full, "Full (LSTM+Transformer+DDM+landmarks)")
print("\nTEST RESULTS - full model (threshold 0.5):")
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()})
print(classification_report(y, yh, target_names=["REAL", "FAKE"], zero_division=0))
pd.DataFrame([res]).round(4).to_csv(os.path.join(CFG["WORK_DIR"], "main_results.csv"), index=False)
sns.heatmap(confusion_matrix(y, yh), annot=True, fmt="d", cmap="Blues", xticklabels=["REAL", "FAKE"], yticklabels=["REAL", "FAKE"])
plt.title("Confusion matrix (test, threshold 0.5)"); plt.xlabel("predicted"); plt.ylabel("true")
plt.savefig(os.path.join(CFG["WORK_DIR"], "confusion.png"), dpi=150); show()
if len(set(y)) > 1:
    fpr, tpr, _ = roc_curve(y, p_full)
    plt.figure(figsize=(4, 4)); plt.plot(fpr, tpr, label=f"AUC = {res['auc']:.3f}"); plt.plot([0, 1], [0, 1], "--", c="gray")
    plt.xlabel("false positive rate"); plt.ylabel("true positive rate"); plt.title("ROC (test)"); plt.legend()
    plt.savefig(os.path.join(CFG["WORK_DIR"], "roc.png"), dpi=150, bbox_inches="tight"); show()
print("main results saved to", os.path.join(CFG["WORK_DIR"], "main_results.csv"), "- ablation starts now")

variants = {"w/o Transformer encoder": dict(use_landmarks=True, use_transformer=False, use_ddm=True),
            "w/o DDM":                 dict(use_landmarks=True, use_transformer=True, use_ddm=False),
            "w/o landmark features":   dict(use_landmarks=False, use_transformer=True, use_ddm=True)}
rows = [res]
for n, fl in variants.items():
    mdl, _ = train_model(re.sub(r"\W+", "_", n), **fl); r, _, _, _ = test_report(mdl, n); rows.append(r)
    pd.DataFrame(rows).set_index("model").round(4).to_csv(os.path.join(CFG["WORK_DIR"], "ablation.csv"))
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
abl = pd.DataFrame(rows).set_index("model").round(4)
cols = ["acc", "bal_acc", "precision", "recall", "f1", "auc"]
print("\nABLATION (test set, threshold 0.5)\n", abl[cols])
print("\nSame models at the validation-tuned threshold\n", abl[["val_tuned_thr"] + [c + "@thr" for c in cols if c != "auc"]])

@torch.no_grad()
def predict_prepared(vid, model=full):
    model.eval(); fr, geo = make_batch([vid], False)
    return torch.sigmoid(model(fr, geo)).item()

for v, t in zip(te.video.head(8), te.label.head(8)):
    pr = predict_prepared(v); print(f"{v:22s} true={'FAKE' if t else 'REAL'}  pred={'FAKE' if pr > .5 else 'REAL'}  P(fake)={pr:.3f}")

print('\nDONE. Outputs (main_results.csv, ablation.csv, curves, confusion.png, roc.png, weights) are in ./work_all')
