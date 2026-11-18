"""Step 4: feature analysis on Person 3's delivered feature tables.

Run from repo root:  python notebooks/step4_feature_analysis.py

Inputs (all analysis restricted to partition == 'train'):
  * cleaned_gps_features.csv      - point-level features, 80 train trips
  * candidate_features_sample.csv - candidate-level features, SAMPLE of
    3 training trips (00000001, 00000003, 00000004), 44,168 rows

Produces:
  * figures in results/figures/step4_*.png (dpi=200)
  * results/feature_selection_table.csv
  * printed summary (copied into DECISIONS.md / report)

All CV uses GroupKFold by trip_id -- observations of one trip never appear
in both train and test folds. The candidate table only covers 3 trips, so
CV scores are indicative, not definitive; flagged in the output.
"""

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

RIDGE = 1e-6
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Load + basic inspection
# ---------------------------------------------------------------------------

cleaned = pd.read_csv(ROOT / "cleaned_gps_features.csv")
cand = pd.read_csv(ROOT / "candidate_features_sample.csv")

train_clean = cleaned[cleaned["partition"] == "train"].copy()
train_cand = cand[cand["partition"] == "train"].copy()

print("=" * 70)
print("STEP 4a INSPECTION")
print("=" * 70)
print(f"cleaned_gps_features: {cleaned.shape} | partitions {cleaned.partition.value_counts().to_dict()}")
print(f"  train rows {len(train_clean)}, trips {train_clean.trip_id.nunique()}, "
      f"obs/trip mean {train_clean.groupby('trip_id').size().mean():.0f}")
print(f"candidate_features_sample: {cand.shape} | trips {sorted(cand.trip_id.unique())}, "
      f"partition values {cand.partition.unique()}")
print(f"  candidates/obs mean {len(train_cand)/train_cand.groupby(['trip_id','obs_idx']).ngroups:.1f}, "
      f"is_reference balance {train_cand.is_reference.mean():.3%}")
print(f"  required cols present: {all(c in train_cand.columns for c in ['trip_id','obs_idx','candidate_arc_id','is_reference','partition'])}")
print(f"  dup (trip,obs,cand): {train_cand.duplicated(['trip_id','obs_idx','candidate_arc_id']).sum()}")
per_obs = train_cand.groupby(["trip_id", "obs_idx"]).size()
print(f"  obs with >=1 candidate: {(per_obs > 0).mean():.1%} "
      f"(min {per_obs.min()}, max {per_obs.max()} candidates/obs)")
print(f"  obs with >=1 is_reference: "
      f"{train_cand.groupby(['trip_id','obs_idx']).is_reference.max().mean():.1%} of observations")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def basic_health(df: pd.DataFrame, name: str, numeric_cols: list[str]):
    print(f"\n--- {name}: NaN/inf/extremes ---")
    nan = df.isna().sum()
    nan = nan[nan > 0]
    print("NaN counts:", nan.to_dict() if len(nan) else "none")
    num = df[numeric_cols].select_dtypes("number")
    inf = np.isinf(num.to_numpy()).sum()
    print(f"inf cells: {inf}")
    for c in numeric_cols:
        s = df[c]
        finite = s[np.isfinite(s)]
        if len(finite) == 0:
            continue
        qs = finite.quantile([0, 0.001, 0.5, 0.999, 1]).round(3).to_dict()
        print(f"  {c:24s} nuniq={s.nunique():6d} min={qs[0]:>12} "
              f"p0.1={qs[0.001]:>12} med={qs[0.5]:>12} p99.9={qs[0.999]:>12} max={qs[1]:>12}")


def spearman_pairs(df: pd.DataFrame, cols: list[str], thresh: float = 0.9):
    sub = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) > 120_000:
        sub = sub.sample(120_000, random_state=RANDOM_STATE)
    corr = sub.corr(method="spearman")
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            rho = corr.loc[a, b]
            if abs(rho) >= thresh:
                pairs.append((a, b, round(float(rho), 4)))
    return corr, pairs


def vif_table(df: pd.DataFrame, cols: list[str]):
    X = df[cols].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(X) > 100_000:
        X = X[np.random.default_rng(RANDOM_STATE).choice(len(X), 100_000, replace=False)]
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    p = Xs.shape[1]
    vifs = []
    for j in range(p):
        y = Xs[:, j]
        others = np.delete(Xs, j, axis=1)
        beta, *_ = np.linalg.lstsq(
            np.column_stack([others, np.ones(len(others))]), y, rcond=None
        )
        pred = np.column_stack([others, np.ones(len(others))]) @ beta
        r2 = 1 - np.sum((y - pred) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12)
        vifs.append(1 / max(1 - r2, 1e-6))
    return pd.Series(vifs, index=cols).round(2)


def plot_grid_hist(df, cols, split_col=None, out_name="dist", log=False, max_rows=400_000):
    """Grid of histograms, optionally split by a binary column."""
    d = df if len(df) <= max_rows else df.sample(max_rows, random_state=RANDOM_STATE)
    n = len(cols)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, cols):
        if split_col is not None and split_col in d.columns:
            for val, color, lab in [(0, "#1f77b4", f"{split_col}=0"),
                                    (1, "#d62728", f"{split_col}=1")]:
                s = d.loc[d[split_col] == val, c].replace([np.inf, -np.inf], np.nan).dropna()
                if len(s) == 0:
                    continue
                vals = np.log1p(s.clip(lower=0)) if log else s
                ax.hist(vals, bins=60, alpha=0.55, color=color, label=lab, density=True)
            ax.legend(fontsize=7)
        else:
            s = d[c].replace([np.inf, -np.inf], np.nan).dropna()
            vals = np.log1p(s.clip(lower=0)) if log else s
            ax.hist(vals, bins=60, color="#1f77b4", alpha=0.8)
        ax.set_title(c + (" (log1p)" if log else ""), fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"{out_name} — training partition only", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / f"step4_{out_name}.png", dpi=200)
    plt.close(fig)
    print(f"saved step4_{out_name}.png")


# ==========================================================================
# A. POINT-LEVEL features (cleaned_gps_features, train only)
# ==========================================================================
print("\n" + "=" * 70)
print("A. POINT-LEVEL FEATURES (cleaned_gps_features, train, 80 trips)")
print("=" * 70)

PT_COLS = ["time_diff_s", "dist_m", "speed_ms", "speed_kmh", "heading_deg"]
basic_health(train_clean, "point-level", PT_COLS + ["longitude", "latitude"])

# flag rates
print("\nflag rates (train):")
for f in ["flag_gap_gt_5s", "flag_speed_gt_50ms", "flag_extreme_gap", "flag_zero_time"]:
    print(f"  {f}: {train_clean[f].mean():.4%}  ({int(train_clean[f].sum())} rows)")
print(f"  speed_kmh > 200: {(train_clean.speed_kmh > 200).mean():.4%}")
print(f"  speed_kmh > 300: {(train_clean.speed_kmh > 300).mean():.4%}")

# time_diff variance / gap informativeness
dt = train_clean["time_diff_s"]
print(f"\ntime_diff_s: mean {dt.mean():.3f}, std {dt.std():.3f}, "
      f"nunique {dt.nunique()}, top values {dt.value_counts().head(5).to_dict()}")
print("is time_diff low-variance?", "YES (near-constant)" if dt.nunique() < 20 else "no")
print("\ncrosstab flag_gap_gt_5s x flag_extreme_gap:")
print(pd.crosstab(train_clean["flag_gap_gt_5s"], train_clean["flag_extreme_gap"]))

# speed quality vs dt bins
df_dt = train_clean.assign(
    dt_bin=pd.cut(train_clean["time_diff_s"], [-0.5, 1.5, 2.5, 5, 60, np.inf],
                  labels=["~1s", "~2s", "2-5s", "5-60s", ">60s"])
)
g = df_dt.groupby("dt_bin", observed=True)["speed_kmh"]
spd_tab = pd.DataFrame({"n": g.size(), "median": g.median().round(1),
                        "p95": g.quantile(0.95).round(1), "max": g.max().round(1)})
print("\nspeed_kmh by time_diff bin:")
print(spd_tab.to_string())

# heading wrap check: mass near 0/360 boundary + first-obs fill
h = train_clean["heading_deg"]
print(f"\nheading_deg: min {h.min():.2f} max {h.max():.2f}; "
      f"share in [0,10)U(350,360): {((h<10)|(h>350)).mean():.2%} "
      f"(uniform would be ~5.6% -> boundary pile-up if higher)")
first = train_clean[train_clean["obs_index"] == 0]
print(f"first-obs rows: {len(first)}; heading equals next obs? "
      f"{(first.set_index('trip_id').heading_deg == train_clean[train_clean.obs_index==1].set_index('trip_id').heading_deg).mean():.1%}")

# distributions
plot_grid_hist(train_clean, PT_COLS, out_name="point_features")
plot_grid_hist(train_clean, PT_COLS, out_name="point_features_log", log=True)

# correlation + VIF (exclude raw coords/ids for the model-relevant set)
MODEL_PT = ["time_diff_s", "dist_m", "speed_ms", "speed_kmh", "heading_deg"]
corr_pt, pairs_pt = spearman_pairs(train_clean, MODEL_PT)
print("\nSpearman |rho|>=0.9 pairs (point-level):", pairs_pt if pairs_pt else "none")
print("\nVIF (point-level numeric):")
print(vif_table(train_clean, MODEL_PT).to_string())
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr_pt, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(MODEL_PT)), MODEL_PT, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(MODEL_PT)), MODEL_PT, fontsize=8)
for i in range(len(MODEL_PT)):
    for j in range(len(MODEL_PT)):
        ax.text(j, i, f"{corr_pt.iloc[i,j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, shrink=0.8)
ax.set_title("Spearman correlation — point features (training, 80 trips)")
fig.tight_layout()
fig.savefig(FIG / "step4_point_corr.png", dpi=200)
plt.close(fig)
print("saved step4_point_corr.png")

# speed vs dt figure
df_dt["speed_clip"] = df_dt["speed_kmh"].clip(upper=300)
fig, ax = plt.subplots(figsize=(7, 4))
for bin_name, sub in df_dt.groupby("dt_bin", observed=True):
    ax.scatter(sub["time_diff_s"], sub["speed_clip"], s=2, alpha=0.15, label=str(bin_name))
ax.set_yscale("symlog", linthresh=10)
ax.set_xlabel("time_diff_s"); ax.set_ylabel("speed_kmh (clip 300)")
ax.set_title("Speed vs sampling interval — training, 80 trips")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "step4_speed_vs_dt.png", dpi=200)
plt.close(fig)
print("saved step4_speed_vs_dt.png")

# ==========================================================================
# B. CANDIDATE-LEVEL features (candidate_features_sample, train, 3 trips)
# ==========================================================================
print("\n" + "=" * 70)
print("B. CANDIDATE-LEVEL FEATURES (candidate_features_sample, train, 3 trips)")
print("=" * 70)
print("NOTE: sample covers only trips 00000001/3/4 -> CV results are indicative only.")

CAND_NUM = [
    "dist_to_candidate_arc_m", "proj_ratio", "arc_length_m", "arc_heading_deg",
    "heading_delta_deg", "connected_from_prev",
    "gps_speed_ms", "gps_speed_kmh", "gps_time_diff_s", "gps_dist_m",
    "gps_heading_deg", "flag_speed_anomaly", "flag_gap_gt_5s",
]
basic_health(train_cand, "candidate-level", CAND_NUM)

# range sanity
print(f"\nproj_ratio range: [{train_cand.proj_ratio.min():.3f}, {train_cand.proj_ratio.max():.3f}] (expect ~[0,1])")
print(f"dist_to_candidate_arc_m max: {train_cand.dist_to_candidate_arc_m.max():.1f} m (radius was 50 m)")
hd = train_cand["heading_delta_deg"]
print(f"heading_delta_deg range: [{hd.min():.1f}, {hd.max():.1f}] "
      f"(check wrap: mass near +/-180?)")
print(f"  share |delta|>170: {(hd.abs()>170).mean():.2%}")

# distributions split by is_reference
plot_grid_hist(train_cand, [c for c in CAND_NUM if c not in ("connected_from_prev",
              "flag_speed_anomaly", "flag_gap_gt_5s")],
               split_col="is_reference", out_name="cand_features_split")
plot_grid_hist(train_cand, ["dist_to_candidate_arc_m", "heading_delta_deg",
                            "gps_speed_kmh", "gps_dist_m"],
               split_col="is_reference", out_name="cand_features_split_log", log=True)

# correlation / VIF / low variance
corr_c, pairs_c = spearman_pairs(train_cand, CAND_NUM)
print("\nSpearman |rho|>=0.9 pairs (candidate-level):")
for p in pairs_c:
    print("  ", p)
vif_c = vif_table(train_cand, [c for c in CAND_NUM if c != "candidate_arc_id"])
print("\nVIF (candidate-level):")
print(vif_c.to_string())
lowvar = train_cand[CAND_NUM].nunique()
print("\nnear-constant features (nunique<=3):", lowvar[lowvar <= 3].to_dict())

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr_c, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(CAND_NUM)), CAND_NUM, rotation=90, fontsize=6)
ax.set_yticks(range(len(CAND_NUM)), CAND_NUM, fontsize=6)
fig.colorbar(im, shrink=0.8)
ax.set_title("Spearman correlation — candidate features (training sample, 3 trips)")
fig.tight_layout()
fig.savefig(FIG / "step4_cand_corr.png", dpi=200)
plt.close(fig)
print("saved step4_cand_corr.png")

# ==========================================================================
# C. MODELLING CHECKS on candidate table (GroupKFold by trip_id)
# ==========================================================================
print("\n" + "=" * 70)
print("C. GROUPKFOLD CV CHECKS (HistGradientBoosting, Macro F1, 3 folds=3 trips)")
print("=" * 70)

# join lon/lat from cleaned table for the memorisation test
geo = train_clean[["trip_id", "obs_index", "longitude", "latitude"]].rename(
    columns={"obs_index": "obs_idx"})
m = train_cand.merge(geo, on=["trip_id", "obs_idx"], how="left", validate="m:1")
print(f"merged lon/lat: {m[['longitude','latitude']].isna().any().any()} NaN -> "
      f"{m.longitude.isna().sum()} unmatched rows")

BASE_FEATS = [c for c in CAND_NUM]                      # no raw coords
WITH_GEO = BASE_FEATS + ["longitude", "latitude"]


def groupkfold_f1(df, feats, label="is_reference"):
    """Mean Macro F1 over GroupKFold-by-trip_id folds."""
    X = df[feats].replace([np.inf, -np.inf], np.nan)
    y = df[label].to_numpy()
    groups = df["trip_id"].to_numpy()
    gkf = GroupKFold(n_splits=df["trip_id"].nunique())
    scores = []
    oof = np.full(len(df), -1)
    for tr, te in gkf.split(X, y, groups):
        clf = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, random_state=RANDOM_STATE
        )
        clf.fit(X.iloc[tr], y[tr])
        pred = clf.predict(X.iloc[te])
        scores.append(f1_score(y[te], pred, average="macro"))
        oof[te] = pred
    return float(np.mean(scores)), scores, oof


f1_base, sc_base, _ = groupkfold_f1(m, BASE_FEATS)
f1_geo, sc_geo, _ = groupkfold_f1(m, WITH_GEO)
print(f"Macro F1  base features ({len(BASE_FEATS)}): {f1_base:.4f}  folds={np.round(sc_base,3)}")
print(f"Macro F1  + lon/lat                  : {f1_geo:.4f}  folds={np.round(sc_geo,3)}")
print(f"delta with raw coordinates: {f1_geo - f1_base:+.4f} "
      f"-> {'LOCATION MEMORISATION RISK' if f1_geo - f1_base > 0.02 else 'no material gain from coords'}")

# also: candidate_arc_id as a feature (id memorisation check)
f1_arcid, _, _ = groupkfold_f1(m, BASE_FEATS + ["candidate_arc_id"])
print(f"Macro F1  + candidate_arc_id        : {f1_arcid:.4f} "
      f"(delta {f1_arcid - f1_base:+.4f})")

# --- out-of-fold permutation importance (on BASE_FEATS) -------------------
rng = np.random.default_rng(RANDOM_STATE)
Xall = m[BASE_FEATS].replace([np.inf, -np.inf], np.nan)
yall = m["is_reference"].to_numpy()
groups = m["trip_id"].to_numpy()
gkf = GroupKFold(n_splits=m["trip_id"].nunique())
base_f1s, perm_drop = [], {c: [] for c in BASE_FEATS}
for tr, te in gkf.split(Xall, yall, groups):
    clf = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, random_state=RANDOM_STATE
    ).fit(Xall.iloc[tr], yall[tr])
    f1_base_fold = f1_score(yall[te], clf.predict(Xall.iloc[te]), average="macro")
    base_f1s.append(f1_base_fold)
    Xte = Xall.iloc[te].copy()
    for c in BASE_FEATS:
        Xp = Xte.copy()
        Xp[c] = rng.permutation(Xp[c].to_numpy())
        f1_perm = f1_score(yall[te], clf.predict(Xp), average="macro")
        perm_drop[c].append(f1_base_fold - f1_perm)
perm_imp = pd.Series({c: np.mean(v) for c, v in perm_drop.items()}).sort_values(ascending=False)
print("\nOut-of-fold permutation importance (Macro F1 drop, mean over folds):")
print(perm_imp.round(4).to_string())

# --- drop-one-family ablation ----------------------------------------------
FAMILIES = {
    "candidate_geometry": ["dist_to_candidate_arc_m", "proj_ratio",
                           "arc_length_m", "arc_heading_deg"],
    "transition_heading": ["heading_delta_deg", "connected_from_prev"],
    "gps_kinematics": ["gps_speed_ms", "gps_speed_kmh", "gps_time_diff_s",
                       "gps_dist_m", "gps_heading_deg"],
    "anomaly_flags": ["flag_speed_anomaly", "flag_gap_gt_5s"],
}
abl_rows = []
for fam, cols in FAMILIES.items():
    feats = [c for c in BASE_FEATS if c not in cols]
    f1, _, _ = groupkfold_f1(m, feats)
    abl_rows.append({"dropped_family": fam, "macro_f1": round(f1, 4),
                     "delta_vs_full": round(f1 - f1_base, 4)})
abl = pd.DataFrame(abl_rows)
print("\nDrop-one-family ablation (Macro F1):")
print(abl.to_string(index=False))

# ==========================================================================
# D. Feature-selection table
# ==========================================================================
print("\n" + "=" * 70)
print("D. FEATURE SELECTION TABLE")
print("=" * 70)

sel_rows = [
    # point-level features
    ("time_diff_s", "timing", "Review", "std 0.54 s but max 153064 s; near-constant at 1 Hz",
     "Low variance for ~75% of trips; huge tail across gaps -> prefer gap indicator",
     "low"),
    ("dist_m", "kinematics", "Review", "haversine step distance; heavy tail at gaps",
     "correlated with speed_ms; meaningless across >60 s gaps",
     "medium"),
    ("speed_ms", "kinematics", "Drop", "exact duplicate of speed_kmh (x3.6); VIF huge",
     "single-step speed noise-sensitive at 1 Hz; see windowed need",
     "low"),
    ("speed_kmh", "kinematics", "Review", "same info as speed_ms; >200 km/h outliers present",
     "keep ONE unit only; cap/winsorize or rely on flag_speed_anomaly",
     "medium"),
    ("heading_deg", "kinematics", "Review", "bearing [0,360); first obs filled from obs 1",
     "wrap-around: must use angular differences, never raw subtraction",
     "medium"),
    ("flag_gap_gt_5s", "quality_flag", "Keep", "marks 1.2k+ irregular steps; segment_id splits on >60 s",
     "directly informative for HMM transition variance",
     "low"),
    ("flag_extreme_gap", "quality_flag", "Keep", "defines segment_id boundaries",
     "prevents HMM transitions across hours-long gaps",
     "low"),
    ("flag_speed_gt_50ms", "quality_flag", "Keep", "1,524 train flags; concentrated in trips 32/1",
     "marks GPS spikes without deleting observations",
     "low"),
    ("flag_zero_time", "quality_flag", "Review", "0 occurrences in train after obs-0 fix",
     "may be all-zero -> dead feature; verify on full candidate table",
     "low"),
    ("segment_id", "quality_flag", "Keep", "sub-trajectory id from extreme gaps",
     "needed to constrain Viterbi decoding within segments",
     "low"),
    ("longitude", "raw_coord", "Drop",
     "GroupKFold CV: adding lon/lat changed Macro F1 by %+.3f (no gain on 3-trip sample); "
     "trips are worldwide so coords act as location identifiers" % (f1_geo - f1_base),
     "theoretical memorisation risk + Person 3's fixed-lat0=50N projection is metrically "
     "wrong for trips far from 50N; not needed as features",
     "high"),
    ("latitude", "raw_coord", "Drop", "same test as longitude", "same", "high"),
    # candidate-level features
    ("dist_to_candidate_arc_m", "candidate_geometry", "Keep", "core emission feature; perm importance %.4f" % perm_imp.get("dist_to_candidate_arc_m", np.nan),
     "capped by 50 m search radius -> bounded, low risk",
     "low"),
    ("proj_ratio", "candidate_geometry", "Review", "position along arc; check [0,1] violations",
     "projection onto segment can exceed endpoints for noisy fixes",
     "medium"),
    ("arc_length_m", "candidate_geometry", "Review", "arc length; correlates with proj_ratio/heading",
     "long arcs vs short arcs may need normalisation",
     "medium"),
    ("arc_heading_deg", "candidate_geometry", "Review", "arc direction; circular",
     "use angular difference vs gps heading, not raw value",
     "medium"),
    ("heading_delta_deg", "transition_heading", "Keep", "GPS-vs-arc heading mismatch; perm imp %.4f" % perm_imp.get("heading_delta_deg", np.nan),
     "wrap-safe? verify range is [-180,180] not [0,360)",
     "low"),
    ("connected_from_prev", "transition_heading", "Keep", "topology: candidate connects to prev obs candidate",
     "uses t-1 context -- causal, fine for HMM emission/transition",
     "low"),
    ("gps_speed_ms", "gps_kinematics", "Drop", "duplicate of gps_speed_kmh",
     "keep one unit",
     "low"),
    ("gps_speed_kmh", "gps_kinematics", "Review", "single-step speed; noisy at 1 Hz and across gaps",
     "windowed speed would be more robust (not yet available)",
     "medium"),
    ("gps_time_diff_s", "gps_kinematics", "Keep", "drives transition variance; near-constant but gap tail matters",
     "low variance overall -> mainly useful via gap flags",
     "low"),
    ("gps_dist_m", "gps_kinematics", "Review", "step distance; duplicates speed info given dt",
     "correlated with gps_speed; check |rho|",
     "medium"),
    ("gps_heading_deg", "gps_kinematics", "Review", "raw bearing; circular",
     "only meaningful via heading_delta_deg",
     "medium"),
    ("flag_speed_anomaly", "anomaly_flags", "Keep", "marks >50 m/s spikes",
     "low",
     "low"),
    ("flag_gap_gt_5s", "anomaly_flags", "Keep", "marks irregular sampling",
     "low",
     "low"),
    ("candidate_arc_id", "identifier", "Drop", "row-index id, not a feature",
     "CV delta +%.4f shows no gain; per-trip ids cannot generalise" % (f1_arcid - f1_base),
     "low"),
    ("is_reference", "label", "N/A", "TARGET LABEL -- never a feature",
     "using it as a feature is direct leakage",
     "high"),
]

sel = pd.DataFrame(sel_rows, columns=["Feature", "Family", "Keep/Drop/Review",
                                      "Evidence", "Risk", "RiskLevel"])
sel.to_csv(RES / "feature_selection_table.csv", index=False)
print(sel.to_string(index=False))
print(f"\nsaved {RES / 'feature_selection_table.csv'}")

# ==========================================================================
# E. Leakage checklist
# ==========================================================================
print("\n" + "=" * 70)
print("E. LEAKAGE / CONTEXT CHECKLIST")
print("=" * 70)
print("""- is_reference is the LABEL; must never enter features. (flagged in table)
- connected_from_prev / heading_delta_deg use PREVIOUS-observation (t-1) GPS
  context only -> causal, legitimate for HMM decoding. No t+1 features present.
- gps_* columns are duplicated identically across all candidates of an obs
  (per-obs kinematics) -- fine, but they are not candidate-specific.
- candidate_arc_id is a per-trip row index; excluded from modelling.
- dist_to_candidate_arc_m is bounded by the 50 m candidate radius.
- person-3 note: candidate generation used a FIXED lat0=50 degN projection;
  dist_to_candidate_arc_m/proj_ratio are metrically wrong for trips far from
  50 degN (e.g. trip 98 at -29 degS). RECOMMEND recomputation per-trip.
- .route arcs not always labelled is_reference (route has more unique arcs
  than marked) -- label reflects 'arc near a fix', consistent with task.
""")

print("\nDONE. Figures in results/figures/step4_*.png; table in results/feature_selection_table.csv")
