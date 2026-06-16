"""
  1. Load raw per-sensor CSVs (~50 Hz), merge onto common time grid
  2. LOF-based outlier cleaning (same approach as OutlierDetection.py)
  3. Window into fixed-length segments
  4. Train & evaluate 1D-CNN and TCN with LOGO outer split
  5. Produce comparison plot

Folder structure expected:
    Subjects/{SubjectName}/processed_data/{Exercise}/session_{N}/
        Accelerometer.csv, Gyroscope.csv, Linear_Accelerometer.csv, ...
"""

import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

# Configuration

BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "Datasets" / "Subjects"
OUTPUT_DIR = BASE_DIR / "Datasets" / "dl_ready"
PLOT_DIR   = BASE_DIR / "plots"

TARGET_HZ  = 50

SENSOR_FILES = {
    "Accelerometer.csv":        {"prefix": "accel",  "cols": ["X", "Y", "Z"]},
    "Gyroscope.csv":            {"prefix": "gyro",   "cols": ["X", "Y", "Z"]},
    "Linear_Accelerometer.csv": {"prefix": "linacc", "cols": ["X", "Y", "Z"]},
}

LABEL_MAP   = {"Pullup": 0, "Pushup": 1, "Squat": 2}
CLASS_NAMES = ["Pullup", "Pushup", "Squat"]
N_CLASSES   = len(LABEL_MAP)

# Window: 400 samples @ 50 Hz = 8 s; stride 200 = 50 % overlap
WINDOW_SIZE = 400
STRIDE      = 200

# LOF 
LOF_N_NEIGHBORS    = 5
LOF_CONTAMINATION  = 0.05
PCA_VARIANCE       = 0.9

# Training
BATCH_SIZE   = 32
MAX_EPOCHS   = 80
PATIENCE     = 10          
LR           = 1e-3
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


# 1. Preprocessing helpers  (mirrors workout_eda.py logic)

_SENSOR_CANDIDATES = {
    "accelerometer": ["Accelerometer.csv"],
    "gyroscope":     ["Gyroscope.csv"],
    "linear_acc":    ["Linear Accelerometer.csv", "Linear_Accelerometer.csv"],
}
_SENSOR_PREFIX = {"accelerometer": "accel", "gyroscope": "gyro", "linear_acc": "linacc"}

_LABEL_ALIASES = {
    "pushups": "Pushup", "pushup": "Pushup",
    "pullups": "Pullup", "pullup": "Pullup", "chinups": "Pullup",
    "squats":  "Squat",  "squat":  "Squat",
}
_SESSION_RE = re.compile(r"^session[_\-\s]*?(\d+)$", re.IGNORECASE)


def _canonical(raw):
    key = re.sub(r"[^a-z0-9]", "", raw.lower())
    return _LABEL_ALIASES.get(key, raw)


def _find_sessions(data_dir):
    """Walk subject/exercise/session_N exactly like workout_eda.find_sessions."""
    rows = []
    for subj_dir in sorted(Path(data_dir).iterdir()):
        if not subj_dir.is_dir():
            continue
        for workout_dir in sorted(subj_dir.iterdir()):
            if not workout_dir.is_dir():
                continue
            exercise = _canonical(workout_dir.name)
            if exercise not in LABEL_MAP:
                continue          # skip processed_data, combined_data, etc.
            for folder in sorted(workout_dir.iterdir()):
                if not folder.is_dir():
                    continue
                m = _SESSION_RE.match(folder.name)
                if not m:
                    continue
                rows.append(dict(subject=subj_dir.name, exercise=exercise,
                                 session=int(m.group(1)), path=folder))
    return rows


def load_session(session_path):
    """
    Load per-sensor CSVs, resample to TARGET_HZ common grid, merge.
    Tries both candidate filenames (e.g. with space vs underscore).
    Returns merged DataFrame or None if any required sensor is missing.
    """
    sensor_dfs = {}
    for sname, candidates in _SENSOR_CANDIDATES.items():
        df = None
        for fname in candidates:
            fp = Path(session_path) / fname
            if fp.exists():
                df = pd.read_csv(fp)
                df.columns = [c.strip().strip('"') for c in df.columns]
                break
        if df is None:
            return None
        sensor_dfs[sname] = df

    t_min = max(d.iloc[:, 0].min() for d in sensor_dfs.values())
    t_max = min(d.iloc[:, 0].max() for d in sensor_dfs.values())
    t_grid = np.arange(t_min, t_max, 1.0 / TARGET_HZ)

    merged = pd.DataFrame({"time": t_grid})
    for sname, df in sensor_dfs.items():
        prefix = _SENSOR_PREFIX[sname]
        t_s = df.iloc[:, 0].values
        for col in df.columns[1:]:
            short = re.sub(r"\s*\(.*\)", "", col).strip()   # "X (m/s^2)" → "X"
            merged[f"{prefix}_{short}"] = np.interp(t_grid, t_s, df[col].values)
    return merged


def clean_outliers_lof(values):
    """LOF outlier detection → interpolate replacements."""
    n, c = values.shape
    if n < LOF_N_NEIGHBORS + 1:
        return values, 0
    X_sc = StandardScaler().fit_transform(values)
    X_pca = PCA(n_components=min(PCA_VARIANCE, c)).fit_transform(X_sc)
    labels = LocalOutlierFactor(
        n_neighbors=LOF_N_NEIGHBORS, contamination=LOF_CONTAMINATION
    ).fit_predict(X_pca)
    mask = labels == -1
    n_out = int(mask.sum())
    if n_out:
        v = values.copy(); v[mask] = np.nan
        v = pd.DataFrame(v).interpolate().ffill().bfill().values
        return v, n_out
    return values, 0


def create_windows(arr, wsize, stride):
    starts = range(0, len(arr) - wsize + 1, stride)
    if not starts:
        return np.empty((0, wsize, arr.shape[1]))
    return np.stack([arr[s:s+wsize] for s in starts])


# 2. Full preprocessing pipeline

def run_preprocessing():
    """Return X, y, groups, session_ids as numpy arrays."""
    cache = OUTPUT_DIR / "X.npy"
    if cache.exists():
        print("[preprocess] Loading cached arrays from", OUTPUT_DIR)
        return (np.load(OUTPUT_DIR / "X.npy"),
                np.load(OUTPUT_DIR / "y.npy"),
                np.load(OUTPUT_DIR / "groups.npy"),
                np.load(OUTPUT_DIR / "session_ids.npy"))

    print("[preprocess] Discovering sessions …")
    sessions = _find_sessions(DATA_DIR)
    subj_names = sorted({r["subject"] for r in sessions})
    sid_map = {name: i for i, name in enumerate(subj_names)}
    print(f"  Found {len(sessions)} sessions across {len(subj_names)} subjects: {subj_names}")

    print("[preprocess] Loading and merging raw sensors …")
    records = []
    for s in sessions:
        merged = load_session(s["path"])
        if merged is None:
            print(f"  [skip] {s['subject']}/{s['exercise']}/session_{s['session']}: missing sensor file")
            continue
        scols = [c for c in merged.columns if c != "time"]
        dur = merged["time"].iloc[-1] - merged["time"].iloc[0]
        records.append(dict(
            subject=s["subject"], subject_id=sid_map[s["subject"]],
            exercise=s["exercise"], session=s["session"],
            sensor_cols=scols,
            values=merged[scols].values.astype(np.float32),
        ))
        print(f"  {s['subject']}/{s['exercise']}/s{s['session']}: "
              f"{len(merged)} samples ({dur:.1f}s)")

    print("[preprocess] LOF cleaning …")
    for r in records:
        r["values"], n = clean_outliers_lof(r["values"])

    print(f"[preprocess] Windowing ({WINDOW_SIZE/TARGET_HZ:.1f}s, "
          f"{STRIDE/TARGET_HZ:.1f}s stride) …")
    aX, ay, ag, asid = [], [], [], []
    for r in records:
        if len(r["values"]) < WINDOW_SIZE:
            print(f"  [skip] {r['subject']}/{r['exercise']} s{r['session']}: "
                  f"only {len(r['values'])} samples < {WINDOW_SIZE}")
            continue
        w = create_windows(r["values"], WINDOW_SIZE, STRIDE)
        lab = LABEL_MAP[r["exercise"]]
        sid = r["subject_id"] * 100 + lab * 10 + r["session"]
        aX.append(w); ay.append(np.full(len(w), lab, dtype=np.int64))
        ag.append(np.full(len(w), r["subject_id"], dtype=np.int64))
        asid.append(np.full(len(w), sid, dtype=np.int64))
        print(f"  {r['subject']}/{r['exercise']} s{r['session']}: {len(w)} windows")

    X = np.concatenate(aX); y = np.concatenate(ay)
    groups = np.concatenate(ag); session_ids = np.concatenate(asid)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, arr in [("X",X),("y",y),("groups",groups),("session_ids",session_ids)]:
        np.save(OUTPUT_DIR / f"{name}.npy", arr)
    print(f"  → X={X.shape}  y={y.shape}")
    return X, y, groups, session_ids



# 3. Model definitions

# ── 1D-CNN ───

class CNN1D(nn.Module):
    """
    Three Conv1D blocks (conv → batchnorm → relu → maxpool → dropout)
    followed by global average pooling and a linear classifier.
    """
    def __init__(self, n_channels, n_classes, dropout=0.3):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(n_channels)   # normalise across subjects
        self.block1 = self._block(n_channels, 64,  kernel=7, dropout=dropout)
        self.block2 = self._block(64,         128, kernel=5, dropout=dropout)
        self.block3 = self._block(128,        256, kernel=3, dropout=dropout)
        self.gap    = nn.AdaptiveAvgPool1d(1)
        self.fc     = nn.Linear(256, n_classes)

    @staticmethod
    def _block(in_ch, out_ch, kernel, dropout):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel//2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_bn(x)   # per-channel normalisation before first conv
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


# ── TCN ───

class TCNResidualBlock(nn.Module):
    """One residual block: two dilated causal convolutions + skip."""
    def __init__(self, n_channels, dilation, kernel_size=3, dropout=0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation   

        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(n_channels, n_channels, kernel_size,
                      padding=pad, dilation=dilation))
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(n_channels, n_channels, kernel_size,
                      padding=pad, dilation=dilation))
        self.drop  = nn.Dropout(dropout)
        self.relu  = nn.ReLU()
        self.pad   = pad   # how many to chop off the right

    def forward(self, x):
        res = x
        out = self.conv1(x)
        if self.pad > 0:
            out = out[:, :, :-self.pad]      # remove future-looking samples
        out = self.relu(self.drop(out))
        out = self.conv2(out)
        if self.pad > 0:
            out = out[:, :, :-self.pad]
        out = self.relu(self.drop(out))
        return self.relu(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network:
    input projection → stacked residual blocks with doubling dilation
    → global average pooling → linear classifier.
    """
    def __init__(self, n_channels, n_classes, hidden=64, n_layers=6,
                 kernel_size=3, dropout=0.2):
        super().__init__()
        self.input_bn   = nn.BatchNorm1d(n_channels)   # normalise across subjects
        self.input_proj = nn.Conv1d(n_channels, hidden, 1)
        self.blocks = nn.Sequential(*[
            TCNResidualBlock(hidden, dilation=2**i,
                             kernel_size=kernel_size, dropout=dropout)
            for i in range(n_layers)
        ])
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = x.transpose(1, 2)       # (B, T, C) → (B, C, T)
        x = self.input_bn(x)        # per-channel normalisation before projection
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


# 4. Training & evaluation helpers

class AugmentedDataset(torch.utils.data.Dataset):
    """
    Training dataset that applies random IMU augmentations on-the-fly.
    Each augmentation mimics a realistic source of between-subject variation:
      - Gaussian noise    : sensor noise & minor calibration differences
      - Time scaling      : different rep speeds across subjects
      - Channel scaling   : phone placement / orientation differences
    Val and test use plain TensorDataset (no augmentation).
    """
    def __init__(self, X, y, noise_std=0.05,
                 time_scale_range=(0.9, 1.1),
                 channel_scale_range=(0.9, 1.1)):
        self.X = X.astype(np.float32)
        self.y = y
        self.noise_std        = noise_std
        self.time_scale_range = time_scale_range
        self.chan_scale_range  = channel_scale_range

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        window = self.X[idx].copy()   # (T, C)
        T, C   = window.shape

        # 1. Gaussian noise
        window += np.random.normal(0, self.noise_std, window.shape)

        # 2. Time scaling — stretch / compress along the time axis
        factor  = np.random.uniform(*self.time_scale_range)
        src_len = max(2, int(T * factor))
        src_idx = np.linspace(0, T - 1, src_len)
        dst_idx = np.linspace(0, src_len - 1, T)
        window  = np.stack([
            np.interp(dst_idx, np.arange(src_len),
                      np.interp(src_idx, np.arange(T), window[:, c]))
            for c in range(C)
        ], axis=1).astype(np.float32)

        # 3. Channel scaling — independent per-axis magnitude shift
        scale  = np.random.uniform(*self.chan_scale_range, size=(1, C))
        window = (window * scale).astype(np.float32)

        return (torch.tensor(window, dtype=torch.float32),
                torch.tensor(self.y[idx], dtype=torch.long))


def make_loaders(X_tr, y_tr, X_val, y_val, X_te, y_te, batch_size):
    """
    Training loader uses AugmentedDataset (random transforms each epoch).
    Val and test loaders use plain TensorDataset — no augmentation.
    """
    def _plain(X, y, shuffle):
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                           torch.tensor(y, dtype=torch.long))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_dl = DataLoader(AugmentedDataset(X_tr, y_tr),
                          batch_size=batch_size, shuffle=True, drop_last=False)
    return train_dl, _plain(X_val, y_val, False), _plain(X_te, y_te, False)


def normalize(X_tr, X_val, X_te):
    """Per-channel z-score: fit on train, apply to val & test."""
    # X shape: (N, T, C) — compute mean/std over N and T
    mean = X_tr.mean(axis=(0, 1), keepdims=True)
    std  = X_tr.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X_tr - mean) / std, (X_val - mean) / std, (X_te - mean) / std


def train_model(model, train_dl, val_dl, max_epochs, patience, lr, device,
                y_train_labels):
    """Train with early stopping on validation loss."""
    model  = model.to(device)
    optim_ = optim.Adam(model.parameters(), lr=lr)

    # Class weights to handle imbalance across exercises and subjects
    cw = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(N_CLASSES),
        y=y_train_labels,
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float32).to(device)
    )

    best_val_loss = float("inf")
    best_state    = None
    wait          = 0
    history       = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, max_epochs + 1):
        # ── train ──
        model.train()
        losses = []
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optim_.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optim_.step()
            losses.append(loss.item())
        tr_loss = np.mean(losses)

        # ── validate ──
        model.eval()
        vlosses, preds, trues = [], [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                vlosses.append(criterion(logits, yb).item())
                preds.append(logits.argmax(1).cpu().numpy())
                trues.append(yb.cpu().numpy())
        val_loss = np.mean(vlosses)
        val_acc  = accuracy_score(np.concatenate(trues), np.concatenate(preds))

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # ── early stopping ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    Early stop at epoch {epoch} "
                      f"(val_loss={val_loss:.4f}, val_acc={val_acc:.3f})")
                break

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}  "
                  f"tr_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"val_acc={val_acc:.3f}")

    model.load_state_dict(best_state)
    return model, history


def evaluate(model, test_dl, device):
    """Return predictions and true labels on a test DataLoader."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(device)
            preds.append(model(xb).argmax(1).cpu().numpy())
            trues.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(trues)


# 5. Outer LOGO loop

def run_experiment(X, y, groups, session_ids, model_name):
    """
    Run one full LOGO experiment for a given model architecture.
    Returns per-fold metrics dict.
    """
    logo = LeaveOneGroupOut()
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_subj = groups[test_idx][0]
        print(f"\n  ── Fold {fold+1}/3  (test subject = {test_subj}) ──")

        X_trainval, y_trainval = X[train_idx], y[train_idx]
        X_test, y_test         = X[test_idx],  y[test_idx]
        sids_trainval          = session_ids[train_idx]

        # Inner validation split: hold out one GroupKFold split
        gkf = GroupKFold(n_splits=3)
        inner_train, inner_val = next(
            gkf.split(X_trainval, y_trainval, sids_trainval)
        )

        X_train, y_train = X_trainval[inner_train], y_trainval[inner_train]
        X_val,   y_val   = X_trainval[inner_val],   y_trainval[inner_val]

        print(f"    Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
        print(f"    Test classes present: "
              f"{[CLASS_NAMES[c] for c in np.unique(y_test)]}")

        # Normalize
        X_train, X_val, X_test = normalize(X_train, X_val, X_test)

        # DataLoaders
        train_dl, val_dl, test_dl = make_loaders(
            X_train, y_train, X_val, y_val, X_test, y_test, BATCH_SIZE
        )

        # Build model
        n_ch = X.shape[2]
        if model_name == "1D-CNN":
            model = CNN1D(n_ch, N_CLASSES)
        elif model_name == "TCN":
            model = TCN(n_ch, N_CLASSES, hidden=64, n_layers=6, kernel_size=3)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Train
        model, history = train_model(
            model, train_dl, val_dl, MAX_EPOCHS, PATIENCE, LR, DEVICE, y_train
        )

        # Evaluate
        preds, trues = evaluate(model, test_dl, DEVICE)

        # Metrics (only over classes present in the test set)
        present = np.unique(trues)
        acc  = accuracy_score(trues, preds)
        f1_m = f1_score(trues, preds, labels=present, average="macro",
                        zero_division=0)

        print(f"    → Accuracy: {acc:.3f}  Macro-F1: {f1_m:.3f}")
        print(classification_report(
            trues, preds, labels=present,
            target_names=[CLASS_NAMES[c] for c in present],
            zero_division=0
        ))

        fold_results.append({
            "fold": fold + 1,
            "test_subject": int(test_subj),
            "accuracy": acc,
            "macro_f1": f1_m,
            "n_test": len(y_test),
            "present_classes": present.tolist(),
        })

    return fold_results


# 6. Comparison plot

def plot_comparison(all_results):
    """
    Bar chart comparing models: mean ± std of accuracy and macro-F1
    across the 3 LOGO folds, similar to the reference screenshot.
    """
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = ["accuracy", "macro_f1"]
    metric_labels = {"accuracy": "Accuracy", "macro_f1": "Macro F1"}
    model_names = list(all_results.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, metric in zip(axes, metrics):
        means, stds = [], []
        for mname in model_names:
            vals = [f[metric] for f in all_results[mname]]
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        x = np.arange(n_models)
        bars = ax.bar(x, means, yerr=stds, capsize=6, width=0.5,
                      color=["#378ADD", "#E24B4A"][:n_models], alpha=0.85,
                      edgecolor="white", linewidth=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=12)
        ax.set_ylabel(metric_labels[metric], fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.set_title(metric_labels[metric], fontsize=13, fontweight="bold")

        # Annotate mean values
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.02,
                    f"{m:.2f}", ha="center", va="bottom", fontsize=11)

    fig.suptitle("1D-CNN vs TCN — LOGO Cross-Validation", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    path = PLOT_DIR / "model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved to {path}")
    plt.close()

    # Also save per-fold detail plot
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    x = np.arange(3)
    width = 0.3
    for i, mname in enumerate(model_names):
        accs = [f["accuracy"] for f in all_results[mname]]
        offset = (i - (n_models-1)/2) * width
        ax2.bar(x + offset, accs, width, label=mname, alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Fold {i+1}\n(test S{all_results[model_names[0]][i]['test_subject']})"
                         for i in range(3)], fontsize=10)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Per-Fold Accuracy", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=11)
    fig2.tight_layout()
    path2 = PLOT_DIR / "per_fold_accuracy.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Per-fold plot saved to {path2}")


# 7. Main

def main():
    print("=" * 60)
    print("HAR Deep Learning Pipeline")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # ── Preprocess ──
    X, y, groups, session_ids = run_preprocessing()

    print(f"\nDataset: {X.shape}")
    for name, cid in LABEL_MAP.items():
        print(f"  {name}: {(y == cid).sum()} windows")
    for sid in np.unique(groups):
        print(f"  Subject {sid}: {(groups == sid).sum()} windows")

    # ── Run experiments ──
    all_results = {}

    for model_name in ["1D-CNN", "TCN"]:
        print("\n" + "═" * 60)
        print(f"  Model: {model_name}")
        print("═" * 60)
        results = run_experiment(X, y, groups, session_ids, model_name)
        all_results[model_name] = results

        # Summary
        accs = [r["accuracy"] for r in results]
        f1s  = [r["macro_f1"] for r in results]
        print(f"\n  {model_name} summary:")
        print(f"    Accuracy:  {np.mean(accs):.3f} ± {np.std(accs):.3f}")
        print(f"    Macro-F1:  {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

    # ── Comparison plot ──
    plot_comparison(all_results)

    # ── Save results ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "dl_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'dl_results.json'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
