import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.signal import find_peaks

pd.set_option("display.max_columns", 50)
plt.rcParams["axes.grid"] = True

DATA_ROOT = "./Datasets/Subjects"
FIGURES_DIR = "./plots"

SENSOR_FILES = {
    "accelerometer": ["Accelerometer.csv"],
    "gyroscope":     ["Gyroscope.csv"],
    "linear_acc":    ["Linear Accelerometer.csv", "Linear_Accelerometer.csv"],
    "barometer":     ["Barometer.csv"],
    "orientation":   ["Orientation.csv"],
}
LABEL_ALIASES = {
    "pushups": "Pushups", "pushup": "Pushups",
    "pullups": "Pullups", "pullup": "Pullups", "chinups": "Pullups",
    "squats":  "Squats",  "squat":  "Squats",
}
SENSOR_PREFIX = {
    "accelerometer": "acc",
    "gyroscope":     "gyr",
    "linear_acc":    "lacc",
    "barometer":     "baro",
    "orientation":   "ori",
}

TARGET_HZ   = 50
TIME_GRAINS = [10, 5, 2]   # window sizes in seconds for the statistics table
FMAX        = 2.5          # upper frequency limit for cadence plot (Hz)
CYCLE_LEN   = 100           # number of points each rep cycle is warped to


def canonical(raw):
    """Map a raw workout-folder name (e.g. 'Pullup', 'pushups') to a
    canonical label ('Pullups', 'Pushups', 'Squats')."""
    key = re.sub(r"[^a-z0-9]", "", raw.lower())
    return LABEL_ALIASES.get(key, raw)


SESSION_DIR_RE = re.compile(r"^session[_\-\s]*?(\d+)$", re.IGNORECASE)

def find_sessions(root):
    """Walk <root>/<subject>/<workout>/<session_N> and build an index.

    Returns a DataFrame with columns: subject, workout, session, path
    where `path` points to the session_N folder containing the CSVs.
    """
    rows = []
    for subj in sorted(os.listdir(root)):
        sp = os.path.join(root, subj)
        if not os.path.isdir(sp):
            continue
        for workout_raw in sorted(os.listdir(sp)):
            wp = os.path.join(sp, workout_raw)
            if not os.path.isdir(wp):
                continue
            workout = canonical(workout_raw)
            for folder in sorted(os.listdir(wp)):
                fpath = os.path.join(wp, folder)
                if not os.path.isdir(fpath):
                    continue
                m = SESSION_DIR_RE.match(folder)
                if m:
                    session_n = int(m.group(1))
                else:
                    # not a "session_N" folder -- skip it (e.g. stray files
                    # or unrecognized sub-folders)
                    continue
                rows.append(dict(subject=subj, workout=workout,
                                  session=session_n, path=fpath))
    return (pd.DataFrame(rows)
              .sort_values(["subject", "workout", "session"])
              .reset_index(drop=True))


def load_all(root):
    sessions = find_sessions(root)
    raw = {}
    for _, s in sessions.iterrows():
        raw[s.path] = {
            name: pd.read_csv(fp).rename(columns=lambda c: c.strip().strip('"'))
            for name, candidates in SENSOR_FILES.items()
            for fp in [next((os.path.join(s.path, f) for f in candidates
                             if os.path.exists(os.path.join(s.path, f))), None)]
            if fp
        }
    return sessions, raw


def hz(df):
    return round(1 / np.median(np.diff(df.iloc[:, 0].to_numpy())), 1)


def concat_sessions(subject, workout, sensor_name, sessions_df, raw_dict):
    """Concatenate all sessions for (subject, workout) for one sensor.

    Returns a single DataFrame with a continuous time axis, or None.
    """
    rows = sessions_df[
        (sessions_df.subject == subject) & (sessions_df.workout == workout)
    ].sort_values("session")

    parts  = []
    offset = 0.0
    for _, s in rows.iterrows():
        d = raw_dict[s.path].get(sensor_name)
        if d is None:
            continue
        shifted = d.copy()
        # reset each session to start at 0, then shift by accumulated offset
        shifted.iloc[:, 0] = d.iloc[:, 0] - d.iloc[0, 0] + offset
        # next session starts one sample-interval after the last sample
        offset = float(shifted.iloc[-1, 0]) + 1.0 / hz(d)
        parts.append(shifted)

    return pd.concat(parts, ignore_index=True) if parts else None


def magnitude(df):
    return np.sqrt((df.iloc[:, 1:4].to_numpy() ** 2).sum(axis=1))


def get(subject, workout, sensor, sessions, raw_combined):
    """Return the combined sensor DataFrame for (subject, workout), or None."""
    row = sessions[(sessions.subject == subject) & (sessions.workout == workout)]
    if row.empty:
        return None
    return raw_combined[row.iloc[0].key].get(sensor)


def window_stats(df, dt_sec):
    t     = df.iloc[:, 0].to_numpy()
    edges = np.arange(t[0], t[-1], dt_sec)
    rows  = []
    for start in edges:
        mask  = (t >= start) & (t < start + dt_sec)
        if mask.sum() < 2:
            continue
        chunk = df.iloc[mask, 1:4] if df.shape[1] >= 4 else df.iloc[mask, 1:2]
        stat  = {}
        for c in chunk.columns:
            short = re.sub(r"\s*\(.*\)", "", c).strip().lower()
            stat[f"{short}_mean"] = chunk[c].mean()
            stat[f"{short}_std"]  = chunk[c].std()
            stat[f"{short}_min"]  = chunk[c].min()
            stat[f"{short}_max"]  = chunk[c].max()
        rows.append(stat)
    return pd.DataFrame(rows)


def stats_for_entry(key, dt_sec, raw_combined):
    parts = []
    for sname, d in raw_combined[key].items():
        if d is None:
            continue
        prefix = SENSOR_PREFIX[sname]
        ws     = window_stats(d, dt_sec)
        if ws.empty:
            continue
        ws.columns = [f"{prefix}_{c}" for c in ws.columns]
        ws.index   = range(len(ws))
        parts.append(ws)
    return pd.concat(parts, axis=1) if parts else pd.DataFrame()


def grained_stats(df, dt_sec):
    t     = df.iloc[:, 0].to_numpy()
    edges = np.arange(t[0], t[-1], dt_sec)
    coarse = []
    for start in edges:
        mask = (t >= start) & (t < start + dt_sec)
        if mask.sum() < 1:
            continue
        chunk = df.iloc[mask, 1:4] if df.shape[1] >= 4 else df.iloc[mask, 1:2]
        coarse.append(chunk.mean().to_dict())

    coarse_df = pd.DataFrame(coarse).dropna()   # drop incomplete intervals

    stat = {}
    for c in coarse_df.columns:
        short = re.sub(r"\s*\(.*\)", "", c).strip().lower()
        stat[f"{short}_mean"] = coarse_df[c].mean()
        stat[f"{short}_std"]  = coarse_df[c].std()
        stat[f"{short}_min"]  = coarse_df[c].min()
        stat[f"{short}_max"]  = coarse_df[c].max()
    return stat


def extract_cycles(df, fs=50.0, n_points=CYCLE_LEN):
    m = magnitude(df)
    peaks, _ = find_peaks(m,
                          distance=int(0.5 * fs),
                          prominence=m.std() * 0.5)
    if len(peaks) < 2:
        return np.empty((0, n_points))
    x_grid = np.linspace(0, 1, n_points)
    cycles = []
    for i in range(len(peaks) - 1):
        seg = m[peaks[i]: peaks[i + 1]]
        if len(seg) < 5:
            continue
        cycles.append(np.interp(x_grid, np.linspace(0, 1, len(seg)), seg))
    return np.array(cycles)


def resample_df(df, grid):
    t = df.iloc[:, 0].to_numpy()
    return {c: np.interp(grid, t, df[c].to_numpy()) for c in df.columns[1:]}


def cadence_spectrum(df, fs=50.0):
    m = magnitude(df) - magnitude(df).mean()
    return signal.welch(m, fs=fs, nperseg=min(1024, len(m)))



def finish_fig(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, f"{name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved {path}")
    plt.show()
    plt.close(fig)



def main():
    # ---- 1. Load all raw sessions -----------------------------------------
    sessions_all, raw_all = load_all(DATA_ROOT)

    if sessions_all.empty:
        raise SystemExit(
            f"No sessions found under DATA_ROOT={DATA_ROOT!r}.\n"
            "Check that DATA_ROOT points to the 'Subjects' folder and that "
            "it contains <subject>/<workout>/session_N/ sub-folders."
        )

    subj_map = {s: f"{i+1:02d}" for i, s in enumerate(sorted(sessions_all.subject.unique()))}
    sessions_all["subject"] = sessions_all["subject"].map(subj_map)

    print(f"Loaded {len(sessions_all)} raw sessions")
    print(sessions_all[["subject", "workout", "session"]].to_string(index=False))
    print("\nSubject name -> ID mapping:")
    for name, sid in subj_map.items():
        print(f"  {name} -> {sid}")

    # ---- 2. Combine sessions per (subject, workout) ------------------------
    sessions_rows = []
    raw_combined  = {}

    for (subj, work), grp in sessions_all.groupby(["subject", "workout"]):
        key = f"{subj}_{work}"
        n_sessions = len(grp)

        raw_combined[key] = {
            sensor: concat_sessions(subj, work, sensor, sessions_all, raw_all)
            for sensor in SENSOR_FILES
        }

        acc = raw_combined[key].get("accelerometer")
        duration = round(acc.iloc[-1, 0] - acc.iloc[0, 0], 1) if acc is not None else None

        sessions_rows.append(dict(
            subject    = subj,
            workout    = work,
            n_sessions = n_sessions,
            duration_s = duration,
            key        = key,
        ))

    sessions = pd.DataFrame(sessions_rows).sort_values(["subject", "workout"]).reset_index(drop=True)
    print("\nCombined sessions (one row per subject x workout):")
    print(sessions.to_string(index=False))

    # ---- 3. Boundary check --------------------------------------------------
    print("\nBoundary check -- gaps > 0.1s in combined recordings:")
    found_gap = False
    for _, s in sessions.iterrows():
        acc = raw_combined[s.key].get("accelerometer")
        if acc is None:
            continue
        t  = acc.iloc[:, 0].to_numpy()
        dt = np.diff(t)
        gaps = np.where(dt > 0.1)[0]
        if len(gaps):
            print(f"  {s.subject} {s.workout}: {len(gaps)} gap(s) at t={t[gaps].round(2)}")
            found_gap = True
    if not found_gap:
        print("  None -- all boundaries are clean.")

    subjects = sorted(sessions.subject.unique())
    works    = sorted(sessions.workout.unique())

    # ---- 4. Session overview -------------------------------------------------
    overview_rows = []
    for _, s in sessions.iterrows():
        acc = raw_combined[s.key].get("accelerometer")
        if acc is not None:
            overview_rows.append(dict(
                subject    = s.subject,
                workout    = s.workout,
                n_sessions = s.n_sessions,
                total_dur_s= s.duration_s,
                n_rows     = len(acc),
                hz         = hz(acc),
            ))
    overview = pd.DataFrame(overview_rows)
    print("\nSession overview:")
    print(overview.to_string(index=False))
    print("\nSessions merged per (subject, workout):")
    print(sessions.pivot_table(index="subject", columns="workout",
                               values="n_sessions", aggfunc="sum", fill_value=0).to_string())

    # ---- 5. Raw time-series per workout -- all subjects ----------------------
    print("\nGenerating raw time-series plots...")
    for sensor, unit in [("accelerometer", "m/s^2"), ("gyroscope", "rad/s"), ("linear_acc", "m/s^2")]:
        fig, axes = plt.subplots(len(works) * 2, len(subjects),
                                 figsize=(5.5 * len(subjects), 2.6 * len(works)),
                                 squeeze=False)
        for wi, work in enumerate(works):
            for si, subj in enumerate(subjects):
                ax_raw = axes[wi * 2,     si]
                ax_mag = axes[wi * 2 + 1, si]
                d = get(subj, work, sensor, sessions, raw_combined)
                if d is not None:
                    t = d.iloc[:, 0].to_numpy()
                    for c in d.columns[1:4]:
                        ax_raw.plot(t, d[c].to_numpy(), lw=0.5, label=c)
                    ax_raw.legend(fontsize=6, ncol=3, loc="upper right")
                    ax_mag.plot(t, magnitude(d), lw=0.6, color="steelblue")
                else:
                    for ax in (ax_raw, ax_mag):
                        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                                transform=ax.transAxes, color="grey")
                ax_raw.set_title(f"subj {subj} -- {work}", fontsize=8)
                ax_raw.set_ylabel(unit, fontsize=7)
                ax_mag.set_ylabel("|mag|", fontsize=7)
                if wi == len(works) - 1:
                    ax_mag.set_xlabel("Time (s)", fontsize=7)
        fig.suptitle(f"{sensor}: axes (odd rows) & magnitude (even rows)", y=1.01, fontsize=10)
        finish_fig(fig, f"01_raw_timeseries_{sensor}")

    # ---- 6. Magnitude distributions -- violin plots --------------------------
    print("\nGenerating violin plots...")
    for sensor, unit in [("linear_acc", "m/s^2"), ("accelerometer", "m/s^2"), ("gyroscope", "rad/s")]:
        rows = []
        for _, s in sessions.iterrows():
            d = raw_combined[s.key].get(sensor)
            if d is None:
                continue
            for v in magnitude(d):
                rows.append({"subject": s.subject, "workout": s.workout, "mag": v})
        df_long = pd.DataFrame(rows)
        if df_long.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.violinplot(data=df_long, x="workout", y="mag",
                       hue="subject", split=len(subjects) == 2,
                       inner="quartile", ax=ax)
        ax.set_title(f"{sensor}: magnitude distribution by workout & subject")
        ax.set_ylabel(f"|{sensor}| ({unit})")
        finish_fig(fig, f"02_violin_{sensor}")

    # ---- 7. Magnitude on normalized time axis ---------------------------------
    print("\nGenerating normalized-time overlays...")
    for sensor in ["linear_acc", "accelerometer"]:
        fig, axes = plt.subplots(len(works), 1,
                                 figsize=(11, 2.8 * len(works)), squeeze=False)
        for ax, work in zip(axes[:, 0], works):
            for subj in subjects:
                d = get(subj, work, sensor, sessions, raw_combined)
                if d is None:
                    continue
                t = d.iloc[:, 0].to_numpy()
                t_norm = (t - t[0]) / (t[-1] - t[0])
                ax.plot(t_norm, magnitude(d), lw=0.6, alpha=0.7, label=f"subj {subj}")
            ax.set_title(work, loc="left", fontsize=9)
            ax.set_ylabel("|mag|", fontsize=8)
            ax.legend(fontsize=7)
        axes[-1, 0].set_xlabel("Normalized time [0, 1]")
        fig.suptitle(f"{sensor}: magnitude on normalized time axis", y=1.01)
        finish_fig(fig, f"03_normalized_time_{sensor}")

    # ---- 8. Rep-cycle overlay (mean +/- std) -----------------------------------
    print("\nGenerating rep-cycle fingerprints...")
    for sensor in ["linear_acc", "accelerometer"]:
        fig, axes = plt.subplots(1, len(works),
                                 figsize=(5 * len(works), 3.5), squeeze=False)
        x = np.linspace(0, 1, CYCLE_LEN)
        for ax, work in zip(axes[0], works):
            for subj in subjects:
                d = get(subj, work, sensor, sessions, raw_combined)
                if d is None:
                    continue
                cycles = extract_cycles(d, fs=hz(d))
                if not len(cycles):
                    continue
                mu, sigma = cycles.mean(axis=0), cycles.std(axis=0)
                ax.plot(x, mu, lw=1.5, label=f"subj {subj} (n={len(cycles)})")
                ax.fill_between(x, mu - sigma, mu + sigma, alpha=0.2)
            ax.set_title(work, fontsize=9)
            ax.set_xlabel("Normalized rep cycle")
            ax.set_ylabel("|mag| (m/s^2)", fontsize=8)
            ax.legend(fontsize=7)
        fig.suptitle(f"{sensor}: mean +/- std rep shape per workout", y=1.01)
        finish_fig(fig, f"04_rep_cycle_{sensor}")

    # ---- 9. Cross-channel correlation -------------------------------------------
    print("\nGenerating cross-channel correlation heatmaps...")
    CORR_SENSORS = ["accelerometer", "gyroscope", "linear_acc"]
    GRID_STEP    = 1.0 / TARGET_HZ

    for subj in subjects:
        subj_works = sorted(sessions[sessions.subject == subj].workout.unique())
        fig, axes = plt.subplots(1, len(subj_works),
                                 figsize=(5.5 * len(subj_works), 5), squeeze=False)
        im = None
        for ax, work in zip(axes[0], subj_works):
            cols = {}
            for sname in CORR_SENSORS:
                d = get(subj, work, sname, sessions, raw_combined)
                if d is None:
                    continue
                prefix = SENSOR_PREFIX[sname]
                grid   = np.arange(d.iloc[0, 0], d.iloc[-1, 0], GRID_STEP)
                for cname, vals in resample_df(d, grid).items():
                    short = re.sub(r"\s*\(.*\)", "", cname).strip().lower()
                    cols[f"{prefix}_{short}"] = vals
            if not cols:
                ax.set_visible(False)
                continue
            min_len = min(len(v) for v in cols.values())
            corr    = pd.DataFrame({k: v[:min_len] for k, v in cols.items()}).corr()
            im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
            ax.set_xticks(range(len(corr)))
            ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
            ax.set_yticks(range(len(corr)))
            ax.set_yticklabels(corr.columns, fontsize=7)
            ax.set_title(work, fontsize=9)
        if im is not None:
            fig.colorbar(im, ax=axes[0], fraction=0.02, pad=0.04)
        fig.suptitle(f"Subject {subj}: channel correlation per workout", fontsize=10)
        finish_fig(fig, f"05_correlation_subj{subj}")

    # ---- 10. Windowed statistics tables ------------------------------------------
    print("\nComputing windowed statistics tables...")
    stat_tables = {}
    for dt in TIME_GRAINS:
        rows = []
        for _, s in sessions.iterrows():
            ws = stats_for_entry(s.key, dt, raw_combined)
            if ws.empty:
                continue
            ws.insert(0, "subject",     s.subject)
            ws.insert(1, "workout",     s.workout)
            ws.insert(2, "window_dt_s", dt)
            rows.append(ws)
        stat_tables[dt] = pd.concat(rows, ignore_index=True)
        print(f"  dt={dt:2d}s -> {len(stat_tables[dt]):4d} windows x {stat_tables[dt].shape[1]} columns")

    # ---- 11. Grained statistics summary --------------------------------------------
    rows = []
    for dt in TIME_GRAINS:
        for _, s in sessions.iterrows():
            row = {"subject": s.subject, "workout": s.workout, "delta_t": dt}
            for sname, d in raw_combined[s.key].items():
                if d is None:
                    continue
                prefix = SENSOR_PREFIX[sname]
                stat   = grained_stats(d, dt)
                row.update({f"{prefix}_{k}": v for k, v in stat.items()})
            rows.append(row)

    grained_summary = pd.DataFrame(rows)

    for work in works:
        print(f"\n-- {work} --------------------------------")
        sub = grained_summary[grained_summary.workout == work]
        acc_cols = [c for c in sub.columns if c.startswith("acc_")][:6]
        print(sub[["subject", "delta_t"] + acc_cols].sort_values(["subject", "delta_t"]).to_string(index=False))

    # ---- 12. Rep cadence (Welch spectrum) -------------------------------------------
    print("\nGenerating cadence spectrum...")
    fig, ax = plt.subplots(figsize=(11, 3.5))
    for work in works:
        d = get(subjects[0], work, "linear_acc", sessions, raw_combined)
        if d is None:
            continue
        f, pxx = cadence_spectrum(d, fs=hz(d))
        band   = f <= FMAX
        fpk    = f[band][np.argmax(pxx[band])]
        ax.semilogy(f[band], pxx[band],
                    label=f"{work}  {fpk:.2f} Hz . {fpk*60:.0f}/min")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("power")
    ax.set_title(f"Subject {subjects[0]}: cadence spectrum")
    ax.legend()
    finish_fig(fig, "06_cadence_spectrum")

    print(f"\nDone. All figures saved to: {os.path.abspath(FIGURES_DIR)}")

    return dict(sessions=sessions, raw_combined=raw_combined,
                 stat_tables=stat_tables, grained_summary=grained_summary)


if __name__ == "__main__":
    main()
