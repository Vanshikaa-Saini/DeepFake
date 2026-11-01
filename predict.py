"""Classify new videos with the trained model.   Usage:  python predict.py video1.mp4 [video2.mp4 ...]
Needs ./work/full.pt (+ the cached features in ./work) produced by deepfake.py. Run from the same folder as deepfake.py.
NOTE: faces are found with dlib's detector (the paper uses FastMTCNN) to avoid an extra dependency."""
import sys, os, re, glob, json, math, random
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F, torchvision, dlib
from sklearn.model_selection import train_test_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

import pandas as pd
CFG = dict(
    # ---------- data ----------
    DATA_ROOT   = None,
    WORK_DIR    = "./work",
    N_FRAMES    = 60,        # PAPER: fixed 60 frames sampled per video
    IMG_SIZE    = 128,       # PAPER: 128x128 input to VGG16
    BALANCE     = True,      # PAPER: equal number of real & fake videos (random under-sampling)
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

CFG.update(json.loads(os.environ.get("DF_OVERRIDES", "{}")))
# ---- command-line options ----
def _opt(name, default):
    if name in sys.argv:
        i = sys.argv.index(name); v = sys.argv[i + 1]; del sys.argv[i:i + 2]; return v
    return default
WORK = _opt("--work", "./work_all" if os.path.isdir("./work_all") else "./work")
MODEL_KEY = _opt("--model", "no_transformer" if os.path.isdir("./work_all") else "full")
THR_ARG = _opt("--thr", None)
MODELS = {  # key: (checkpoint, row name in ablation.csv, flags)
    "full":           ("full.pt", "Full (LSTM+Transformer+DDM+landmarks)", dict(use_landmarks=True, use_transformer=True, use_ddm=True)),
    "no_transformer": ("w_o_Transformer_encoder.pt", "w/o Transformer encoder", dict(use_landmarks=True, use_transformer=False, use_ddm=True)),
    "no_ddm":         ("w_o_DDM.pt", "w/o DDM", dict(use_landmarks=True, use_transformer=True, use_ddm=False)),
    "no_landmarks":   ("w_o_landmark_features.pt", "w/o landmark features", dict(use_landmarks=False, use_transformer=True, use_ddm=True)),
}
assert MODEL_KEY in MODELS, f"--model must be one of {list(MODELS)}"
_lm = os.path.join(WORK, "shape_predictor_68_face_landmarks.dat")
if not os.path.exists(_lm): _lm = "./work/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(_lm)
detector = dlib.get_frontal_face_detector()
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
        self.att = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
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
        vgg = torchvision.models.vgg16(weights=None)
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
            f = F.adaptive_avg_pool2d(self.cnn(x), 1).flatten(1).view(B, T, -1)     # (B,T,512)
        if self.use_lm: f = torch.cat([f, geo], -1)
        h, _ = self.lstm(f)                                                         # H_LSTM (B,T,K)
        z = self.tf(self.pos(h))                                                    # X = H + PE -> Transformer
        v = ddm(z, CFG["DDM_MODE"]) if self.use_ddm else z.mean(1)
        return self.head(v).squeeze(1)                                              # logit; sigmoid applied in loss / at inference


# ---- recover the exact z-score statistics used in training (same cache, same seeded split) ----
cache = sorted(glob.glob(os.path.join(WORK, f"prep_T{CFG['N_FRAMES']}_S{CFG['IMG_SIZE']}_*.npz")), key=os.path.getmtime)
assert cache, "no cached features found in ./work - run deepfake.py first (same N_FRAMES / IMG_SIZE)"
z = np.load(cache[-1], allow_pickle=True); names = list(z["names"]); geo_all = z["geo"]
df = pd.DataFrame(dict(video=names, label=[int(n.startswith("FAKE")) for n in names]))
trv, te = train_test_split(df, test_size=CFG["TEST_FRAC"], stratify=df.label, random_state=SEED if "SEED" in globals() else 42)
tr, va = train_test_split(trv, test_size=CFG["VAL_FRAC"], stratify=trv.label, random_state=42)
g_tr = np.concatenate([geo_all[i] for i in tr.index]); G_MEAN, G_STD = g_tr.mean(0), g_tr.std(0) + 1e-6

_ck, _row, _flags = MODELS[MODEL_KEY]
model = DeepfakeNet(**_flags).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(WORK, _ck), map_location=DEVICE)); model.eval()
THR = 0.5
if THR_ARG is not None: THR = float(THR_ARG)
else:
    try:
        _abl = pd.read_csv(os.path.join(WORK, "ablation.csv")).set_index("model")
        if "val_tuned_thr" in _abl.columns and _row in _abl.index: THR = float(_abl.loc[_row, "val_tuned_thr"])
    except Exception: pass
print(f"model: {MODEL_KEY} ({WORK}/{_ck}) | decision threshold: {THR:.4f}")

try:
    from facenet_pytorch import MTCNN
    _mtcnn = MTCNN(keep_all=True, device="cpu", min_face_size=20, thresholds=[0.6, 0.7, 0.7])   # same detector family as the paper (MTCNN)
except Exception as e:
    _mtcnn = None; print("  (MTCNN not available -> using dlib face detector only:", type(e).__name__, ")")

def _find_box(f):
    """largest face box (x1,y1,x2,y2) in frame coords, or None. MTCNN first, then dlib."""
    sc = 960 / f.shape[1] if f.shape[1] > 960 else 1.0
    small = cv2.resize(f, None, fx=sc, fy=sc) if sc != 1.0 else f
    if _mtcnn is not None:
        try:
            boxes, probs = _mtcnn.detect(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            if boxes is not None and len(boxes):
                k = int(np.argmax([(float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1])) for b in boxes]))
                return [float(v) / sc for v in boxes[k]]
        except Exception: pass
    rects = detector(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 1)
    if len(rects):
        r = max(rects, key=lambda r: r.area()); return [v / sc for v in (r.left(), r.top(), r.right(), r.bottom())]
    return None

def crop_faces(path, max_detect=120):
    cap = cv2.VideoCapture(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); frames = []
    want = set(np.linspace(0, max(n - 1, 0), min(max_detect, max(n, 1))).astype(int).tolist()); i = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if i in want: frames.append(f)
        i += 1
    cap.release(); print(f"  read {len(frames)} frames to search for faces (video has {n} frames)")
    if not frames: return []
    os.makedirs(WORK, exist_ok=True); cv2.imwrite(os.path.join(WORK, "debug_frame.jpg"), frames[len(frames) // 2])
    print("  saved a sample frame to work/debug_frame.jpg - open it to check the face is visible and upright")
    # pick the rotation that yields the most faces on a few probe frames (handles sideways phone videos)
    rots = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    probe = frames[:: max(1, len(frames) // 8)]
    best_rot, best_cnt = 0, -1
    for deg, code in rots.items():
        cnt = sum(_find_box(p if code is None else cv2.rotate(p, code)) is not None for p in probe)
        if cnt > best_cnt: best_rot, best_cnt = deg, cnt
        if deg == 0 and cnt >= len(probe) // 2: break
    code = rots[best_rot]
    if best_rot: print(f"  video looks rotated; using {best_rot} degree rotation")
    faces = []
    for f in frames:
        if code is not None: f = cv2.rotate(f, code)
        b = _find_box(f)
        if b is None: continue
        x1, y1, x2, y2 = b; w, h = x2 - x1, y2 - y1; m = 0.15
        x1, y1, x2, y2 = [max(int(v), 0) for v in (x1 - m * w, y1 - m * h, x2 + m * w, y2 + m * h)]
        face = f[y1:y2, x1:x2]
        if face.size and min(face.shape[:2]) > 20: faces.append(face)
    return faces

@torch.no_grad()
def predict(path):
    faces = crop_faces(path)
    print(f"  faces found: {len(faces)}")
    if len(faces) == 0: return None, 0
    if len(faces) < 10: print("  WARNING: very few faces found - result is unreliable")
    T, S = CFG["N_FRAMES"], CFG["IMG_SIZE"]
    faces = [faces[i] for i in np.linspace(0, len(faces) - 1, T).round().astype(int)]
    fr, geo = [], []
    for f in faces:
        e = cv2.resize(enhance(cv2.resize(f, (S, S))), (S, S), interpolation=cv2.INTER_AREA)
        geo.append(geometric_features(landmarks68(e))); fr.append(cv2.cvtColor(e, cv2.COLOR_BGR2RGB))
    fr = torch.from_numpy(np.stack(fr))[None].to(DEVICE).permute(0, 1, 4, 2, 3).float() / 255.
    g = torch.from_numpy((np.stack(geo) - G_MEAN) / G_STD)[None].float().to(DEVICE)
    return torch.sigmoid(model(fr, g)).item(), len(faces)

if __name__ == "__main__":
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not paths: sys.exit("usage: python predict.py video.mp4 [more.mp4 ...]")
    for p in paths:
        pr, n = predict(p)
        if pr is None: print(f"{os.path.basename(p)}: no usable face found ({n} faces detected)"); continue
        print(f"{os.path.basename(p)}: P(fake) = {pr:.4f}  (threshold {THR:.4f})  ->  {'FAKE' if pr > THR else 'REAL'}")