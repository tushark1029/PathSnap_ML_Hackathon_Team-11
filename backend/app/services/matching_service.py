"""Map-matching prediction service for the evaluator API.

Wraps the team's trained candidate-ranking models + HMM/Viterbi decoder
(see train_advanced_features_rf.py and road_network_hmm_decoder.py at the
repo root) so the FastAPI layer can serve predictions for the ten
reserved evaluator trips.

Design notes:
- Ground-truth `.route` files are read ONLY to compute the accuracy
  metric server-side. They are never returned to the client and no
  label-derived feature is ever fed to the model (`is_reference` is not
  in ADVANCED_FEATURE_COLS).
- The candidate-generation loop is a faithful port of
  build_enriched_candidates() minus the ground-truth alignment block.
- Models are loaded lazily once and cached for the process lifetime.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Repo root = backend/app/services/matching_service.py -> parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Directory containing the reserved evaluator trips. Each trip is a
# sub-directory <trip_id>/ holding <trip_id>.{track,nodes,arcs,route}.
# Override with PATHSNAP_EVALUATOR_DIR (e.g. on the API laptop).
EVALUATOR_DIR = Path(
    os.environ.get("PATHSNAP_EVALUATOR_DIR", PROJECT_ROOT / "evaluator_test")
)

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_RF = MODELS_DIR / "best_model_advanced_rf.joblib"
MODEL_HGB = MODELS_DIR / "best_model_advanced_hgb.joblib"

# Must match train_advanced_features_rf.py exactly — order matters because
# the models were fit on these columns positionally.
ADVANCED_FEATURE_COLS = [
    "dist_to_candidate_arc_m", "proj_ratio", "arc_length_m", "arc_heading_deg",
    "heading_delta_deg", "connected_from_prev", "gps_speed_kmh", "gps_dist_m",
    "gps_time_diff_s", "flag_speed_anomaly", "flag_gap_gt_5s",
    "cos_heading_diff", "dist_rank", "dist_ratio_to_min", "dist_diff_from_min",
    "dist_to_u_m", "dist_to_v_m",
]

RADIUS_M = 50.0          # candidate search radius, same as training
BLEND_W = {"rf": 0.60, "hgb": 0.40}  # ensemble weights from training script

_models = {}


def _load_models():
    if not _models:
        _models["rf"] = joblib.load(MODEL_RF)
        _models["hgb"] = joblib.load(MODEL_HGB)
    return _models


def _trip_dir(trip_id: str) -> Path:
    # trip_id is validated by the route layer (alphanumeric only), but
    # guard again here so this service is safe to call directly.
    if not trip_id or not all(c.isalnum() for c in trip_id):
        raise ValueError(f"Invalid trip_id: {trip_id!r}")
    return EVALUATOR_DIR / trip_id


def list_evaluator_trips() -> list[dict]:
    """Return the reserved evaluator trips with observation counts.

    Only IDs and counts are exposed — never any route/label data.
    """
    trips = []
    if not EVALUATOR_DIR.is_dir():
        return trips
    for d in sorted(EVALUATOR_DIR.iterdir()):
        if not d.is_dir():
            continue
        track = d / f"{d.name}.track"
        if track.is_file():
            # count non-empty lines cheaply
            with open(track, "rb") as f:
                n = sum(1 for line in f if line.strip())
            trips.append({"trip_id": d.name, "observations": n})
    return trips


def _read_trip(trip_id: str):
    td = _trip_dir(trip_id)
    tid = td.name
    track_p = td / f"{tid}.track"
    nodes_p = td / f"{tid}.nodes"
    arcs_p = td / f"{tid}.arcs"
    route_p = td / f"{tid}.route"
    for p in (track_p, nodes_p, arcs_p):
        if not p.is_file():
            raise FileNotFoundError(f"Missing required file: {p.name}")

    track = pd.read_csv(track_p, sep=r"\s+", header=None,
                        names=["lon", "lat", "time"])
    nodes = pd.read_csv(nodes_p, sep=r"\s+", header=None, names=["lon", "lat"])
    arcs = pd.read_csv(arcs_p, sep=r"\s+", header=None, names=["u", "v"])
    route = (pd.read_csv(route_p, header=None, names=["arc_id"])
             if route_p.is_file() else None)
    return track, nodes, arcs, route


def _build_candidates(track, nodes, arcs, radius_m=RADIUS_M) -> pd.DataFrame:
    """Generate candidate arcs + ranking features per GPS observation.

    Port of build_enriched_candidates() from train_advanced_features_rf.py
    with the ground-truth alignment block removed (no labels used).
    """
    n_obs = len(track)
    lat0 = float(np.mean(nodes["lat"]))
    m_per_deg_lat = 111139.0
    m_per_deg_lon = 111139.0 * np.cos(np.radians(lat0))

    node_x = nodes["lon"].to_numpy() * m_per_deg_lon
    node_y = nodes["lat"].to_numpy() * m_per_deg_lat

    u_idx = arcs["u"].to_numpy()
    v_idx = arcs["v"].to_numpy()
    ax1, ay1 = node_x[u_idx], node_y[u_idx]
    ax2, ay2 = node_x[v_idx], node_y[v_idx]
    adx, ady = ax2 - ax1, ay2 - ay1
    arc_lengths = np.hypot(adx, ady)
    arc_headings = (np.degrees(np.arctan2(adx, ady)) + 360.0) % 360.0

    # Spatial grid index over arc bounding boxes (same as training).
    cell = float(radius_m)
    grid = {}
    min_x, max_x = np.minimum(ax1, ax2), np.maximum(ax1, ax2)
    min_y, max_y = np.minimum(ay1, ay2), np.maximum(ay1, ay2)
    for aid in range(len(arcs)):
        for cx in range(int(min_x[aid] // cell), int(max_x[aid] // cell) + 1):
            for cy in range(int(min_y[aid] // cell), int(max_y[aid] // cell) + 1):
                grid.setdefault((cx, cy), []).append(aid)

    tx = track["lon"].to_numpy() * m_per_deg_lon
    ty = track["lat"].to_numpy() * m_per_deg_lat
    ttime = track["time"].to_numpy()

    dt = np.zeros(n_obs)
    dt[1:] = ttime[1:] - ttime[:-1]
    pdx = np.zeros(n_obs)
    pdy = np.zeros(n_obs)
    pdx[1:] = tx[1:] - tx[:-1]
    pdy[1:] = ty[1:] - ty[:-1]
    dist_m = np.hypot(pdx, pdy)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed_ms = np.where(dt > 0, dist_m / dt, 0.0)
    gps_headings = (np.degrees(np.arctan2(pdx, pdy)) + 360.0) % 360.0
    if n_obs > 1:
        gps_headings[0] = gps_headings[1]

    rows = []
    prev_cand_arcs = set()
    prev_cand_nodes_v = set()

    for i in range(n_obs):
        px, py = tx[i], ty[i]
        cx1 = int((px - radius_m) // cell)
        cx2 = int((px + radius_m) // cell)
        cy1 = int((py - radius_m) // cell)
        cy2 = int((py + radius_m) // cell)
        cands = set()
        for cx in range(cx1, cx2 + 1):
            for cy in range(cy1, cy2 + 1):
                if (cx, cy) in grid:
                    cands.update(grid[(cx, cy)])

        raw = []
        for aid in cands:
            x1v, y1v, x2v, y2v = ax1[aid], ay1[aid], ax2[aid], ay2[aid]
            dx, dy = x2v - x1v, y2v - y1v
            L2 = dx * dx + dy * dy
            if L2 == 0:
                t_val, d_val = 0.0, float(np.hypot(px - x1v, py - y1v))
            else:
                t_val = max(0.0, min(1.0, ((px - x1v) * dx + (py - y1v) * dy) / L2))
                d_val = float(np.hypot(px - (x1v + t_val * dx),
                                       py - (y1v + t_val * dy)))
            if d_val > radius_m:
                continue
            h_diff = abs(gps_headings[i] - arc_headings[aid]) % 360.0
            if h_diff > 180.0:
                h_diff = 360.0 - h_diff
            is_connected = 1 if (aid in prev_cand_arcs
                                 or u_idx[aid] in prev_cand_nodes_v
                                 or i == 0) else 0
            d_u = float(np.hypot(px - x1v, py - y1v))
            d_v = float(np.hypot(px - x2v, py - y2v))
            raw.append({
                "obs_idx": i,
                "candidate_arc_id": aid,
                "dist_to_candidate_arc_m": d_val,
                "proj_ratio": t_val,
                "arc_length_m": float(arc_lengths[aid]),
                "arc_heading_deg": float(arc_headings[aid]),
                "heading_delta_deg": float(h_diff),
                "connected_from_prev": is_connected,
                "gps_speed_kmh": float(speed_ms[i] * 3.6),
                "gps_time_diff_s": float(dt[i]),
                "gps_dist_m": float(dist_m[i]),
                "flag_speed_anomaly": 1 if speed_ms[i] > 50.0 else 0,
                "flag_gap_gt_5s": 1 if dt[i] > 5.0 else 0,
                "cos_heading_diff": float(np.cos(np.radians(h_diff))),
                "dist_to_u_m": d_u,
                "dist_to_v_m": d_v,
            })

        if raw:
            raw.sort(key=lambda c: c["dist_to_candidate_arc_m"])
            dmin = raw[0]["dist_to_candidate_arc_m"]
            for rank, c in enumerate(raw):
                c["dist_rank"] = rank + 1
                c["dist_ratio_to_min"] = c["dist_to_candidate_arc_m"] / (dmin + 0.1)
                c["dist_diff_from_min"] = c["dist_to_candidate_arc_m"] - dmin

        prev_cand_arcs = {c["candidate_arc_id"] for c in raw}
        prev_cand_nodes_v = {v_idx[c["candidate_arc_id"]] for c in raw}
        rows.extend(raw)

    return pd.DataFrame(rows)


def _viterbi_decode(cands: pd.DataFrame, arcs) -> list[int]:
    """HMM Viterbi decoding with backpointers.

    Same transition model as the training evaluation (run_viterbi in
    train_advanced_features_rf.py): same-arc 0.0, connected -0.3,
    disconnected -4.0; emission = log(prob) - 0.5*(dist/15)^2.
    Backpointers are used instead of copying path lists, so memory is
    O(T * candidates) rather than O(T^2).
    """
    eps = 1e-6
    sigma = 15.0
    u_arr = arcs["u"].to_numpy()
    v_arr = arcs["v"].to_numpy()

    groups = [g for _, g in cands.groupby("obs_idx", sort=True)]
    if not groups:
        return []

    def emission(g):
        return (np.log(g["prob"].to_numpy() + eps)
                - 0.5 * (g["dist_to_candidate_arc_m"].to_numpy() / sigma) ** 2)

    arcs0 = groups[0]["candidate_arc_id"].to_numpy()
    dp = {int(a): float(e) for a, e in zip(arcs0, emission(groups[0]))}
    # backpointers[t] maps curr_arc -> prev_arc chosen at step t
    back = []

    for t in range(1, len(groups)):
        g = groups[t]
        curr_arcs = g["candidate_arc_id"].to_numpy()
        curr_em = emission(g)
        new_dp = {}
        bp_t = {}
        for c_idx, curr_arc in enumerate(curr_arcs):
            curr_arc = int(curr_arc)
            curr_u = u_arr[curr_arc]
            e = float(curr_em[c_idx])
            best_score, best_prev = -np.inf, None
            for prev_arc, prev_score in dp.items():
                if prev_arc == curr_arc:
                    trans = 0.0
                elif v_arr[prev_arc] == curr_u:
                    trans = -0.3
                else:
                    trans = -4.0
                s = prev_score + trans + e
                if s > best_score:
                    best_score, best_prev = s, prev_arc
            if best_prev is not None:
                new_dp[curr_arc] = best_score
                bp_t[curr_arc] = best_prev
        if not new_dp:  # no candidates survived; restart chain
            new_dp = {int(a): float(e) for a, e in zip(curr_arcs, curr_em)}
            bp_t = {int(a): -1 for a in curr_arcs}
        dp = new_dp
        back.append(bp_t)

    # Backtrack from the best final state.
    final_arc = max(dp, key=lambda k: dp[k])
    path = [final_arc]
    for t in range(len(back) - 1, -1, -1):
        prev = back[t].get(path[-1], -1)
        if prev == -1:
            break
        path.append(prev)
    path.reverse()
    return path


def _align_route_to_obs(track, nodes, arcs, route) -> np.ndarray:
    """Map each GPS observation to its nearest reference-route arc.

    Same windowed alignment as the training/eval scripts. Server-side
    only — used solely for the accuracy metric.
    """
    ref_arcs = route["arc_id"].to_numpy()
    R = len(ref_arcs)
    lat0 = float(np.mean(nodes["lat"]))
    m_per_deg_lat = 111139.0
    m_per_deg_lon = 111139.0 * np.cos(np.radians(lat0))
    node_x = nodes["lon"].to_numpy() * m_per_deg_lon
    node_y = nodes["lat"].to_numpy() * m_per_deg_lat
    tx = track["lon"].to_numpy() * m_per_deg_lon
    ty = track["lat"].to_numpy() * m_per_deg_lat

    ru = arcs.iloc[ref_arcs]["u"].to_numpy()
    rv = arcs.iloc[ref_arcs]["v"].to_numpy()
    rx1, ry1 = node_x[ru], node_y[ru]
    rx2, ry2 = node_x[rv], node_y[rv]
    rdx, rdy = rx2 - rx1, ry2 - ry1
    rL2 = rdx * rdx + rdy * rdy

    true = np.empty(len(track), dtype=np.int64)
    cur = 0
    for i in range(len(track)):
        px, py = tx[i], ty[i]
        w0, w1 = max(0, cur - 20), min(R, cur + 60)
        best_d, best_i = np.inf, cur
        for r in range(w0, w1):
            L2 = rL2[r]
            t = 0.0 if L2 == 0 else max(0.0, min(1.0,
                ((px - rx1[r]) * rdx[r] + (py - ry1[r]) * rdy[r]) / L2))
            d = np.hypot(px - (rx1[r] + t * rdx[r]),
                         py - (ry1[r] + t * rdy[r]))
            if d < best_d:
                best_d, best_i = d, r
        if best_d > 45.0:  # fallback: search whole route
            for r in range(R):
                L2 = rL2[r]
                t = 0.0 if L2 == 0 else max(0.0, min(1.0,
                    ((px - rx1[r]) * rdx[r] + (py - ry1[r]) * rdy[r]) / L2))
                d = np.hypot(px - (rx1[r] + t * rdx[r]),
                             py - (ry1[r] + t * rdy[r]))
                if d < best_d:
                    best_d, best_i = d, r
        cur = best_i
        true[i] = ref_arcs[best_i]
    return true


def _route_coords(pred_arcs: list[int], nodes, arcs) -> list[list[float]]:
    """Convert the predicted per-observation arc sequence into a road
    polyline of [lon, lat] points by walking arc endpoints (u -> v)."""
    u_arr = arcs["u"].to_numpy()
    v_arr = arcs["v"].to_numpy()
    lon = nodes["lon"].to_numpy()
    lat = nodes["lat"].to_numpy()
    coords = []
    prev_end = None
    for aid in pred_arcs:
        p1 = (float(lon[u_arr[aid]]), float(lat[u_arr[aid]]))
        p2 = (float(lon[v_arr[aid]]), float(lat[v_arr[aid]]))
        if prev_end is None or p1 != prev_end:
            coords.append([p1[0], p1[1]])
        coords.append([p2[0], p2[1]])
        prev_end = p2
    return coords


def predict_trip(trip_id: str) -> dict:
    """Run the full map-matching pipeline for one evaluator trip.

    Returns a dict ready to be JSON-serialised by the route layer.
    Raises FileNotFoundError / ValueError on bad input.
    """
    t_start = time.time()

    track, nodes, arcs, route = _read_trip(trip_id)

    # 1. Candidate generation
    t0 = time.time()
    cands = _build_candidates(track, nodes, arcs)
    t_cands = time.time() - t0
    if cands.empty:
        raise ValueError(f"No candidate arcs found for trip {trip_id}")

    # 2. Model inference (champion blend: 0.6 RF + 0.4 HGB)
    t0 = time.time()
    models = _load_models()
    X = cands[ADVANCED_FEATURE_COLS].to_numpy()
    probs = (BLEND_W["rf"] * models["rf"].predict_proba(X)[:, 1]
             + BLEND_W["hgb"] * models["hgb"].predict_proba(X)[:, 1])
    cands["prob"] = probs
    t_model = time.time() - t0

    # 3. HMM Viterbi decoding -> one arc per observation
    t0 = time.time()
    pred_arcs = _viterbi_decode(cands, arcs)
    t_decode = time.time() - t0

    # Classifier-only baseline (top-1 per observation) for comparison.
    top1 = (cands.loc[cands.groupby("obs_idx")["prob"].idxmax()]
            ["candidate_arc_id"].to_numpy())

    # 4. Route geometry for the map
    route_coords = _route_coords(pred_arcs, nodes, arcs)

    # 5. Metrics vs ground truth (server-side only)
    metrics = {}
    if route is not None and len(route):
        true_arcs = _align_route_to_obs(track, nodes, arcs, route)
        # Predictions are one arc per candidate-group (obs_idx); align them
        # back to observation indices before comparing with ground truth.
        group_obs = np.array(
            [int(g) for g, _ in cands.groupby("obs_idx", sort=True)])
        pred_by_obs = dict(zip(group_obs.tolist(), pred_arcs))
        top1_by_obs = dict(zip(group_obs.tolist(), top1.tolist()))
        common = np.intersect1d(group_obs, np.arange(len(track)))
        pred_aligned = np.array([pred_by_obs[int(i)] for i in common])
        top1_aligned = np.array([top1_by_obs[int(i)] for i in common])
        true_aligned = true_arcs[common]
        metrics["hmm_accuracy"] = round(
            float(np.mean(pred_aligned == true_aligned)), 4)
        metrics["classifier_accuracy"] = round(
            float(np.mean(top1_aligned == true_aligned)), 4)

    v_arr = arcs["v"].to_numpy()
    u_arr = arcs["u"].to_numpy()
    cont = 0
    for a, b in zip(pred_arcs, pred_arcs[1:]):
        if a == b or v_arr[a] == u_arr[b]:
            cont += 1
    metrics["route_continuity"] = round(
        cont / max(1, len(pred_arcs) - 1), 4)
    metrics["observations"] = int(len(track))
    metrics["candidate_rows"] = int(len(cands))
    metrics["mean_candidates_per_obs"] = round(
        float(len(cands) / max(1, cands["obs_idx"].nunique())), 2)

    timings = {
        "candidate_generation_s": round(t_cands, 3),
        "model_inference_s": round(t_model, 3),
        "viterbi_decode_s": round(t_decode, 3),
        "total_s": round(time.time() - t_start, 3),
    }

    return {
        "trip_id": trip_id,
        "status": "success",
        "predicted_route": route_coords,
        "gps_observations": [
            {"longitude": float(r.lon), "latitude": float(r.lat),
             "timestamp": float(r.time)}
            for r in track.itertuples()
        ],
        "metrics": metrics,
        "timings": timings,
    }
