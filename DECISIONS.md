# PathSnap — decisions, assumptions and data-quality log

Person 4 scope: feature analysis and visualisation. All analysis uses the
**training partition only** (80 trips) unless stated otherwise.

## Data layout (verified by inspection, Step 0)

- Workspace root contains `Student folder/Student folder/{train,student_test}/<trip_id>/`.
- Each trip folder contains four files named `<trip_id>.{track,nodes,arcs,route}`.
- **Deviation from brief:** there is NO single shared road network. Each trip
  ships its own `.nodes`/`.arcs` pair (a per-trip clipped network; median
  ~9.6k nodes / ~19.5k arcs, max ~155k nodes / ~307k arcs). Total across the
  80 training trips ≈ 1.9M nodes / 3.9M arcs, which is presumably where the
  brief's numbers came from.
- **Deviation from brief:** the "evaluator-test" partition is not present in
  this workspace. Only `train` (80 trips) and `student_test` (10 trips) exist.
- File formats confirmed: `.track` = lon, lat, t (whitespace-separated, no
  header); `.nodes` = lon, lat; `.arcs` = from_node, to_node (0-based ids,
  row index = arc id); `.route` = 1-D sequence of arc ids.

## Coordinate / distance decisions

- Trips are **worldwide** (Stockholm, St. Petersburg, Tokyo, Mexico City,
  etc.), not a single region — verified across all 80 training tracks.
- Distance: **haversine** (mean Earth radius 6 371 000 m), vectorised numpy,
  in `src/data/loading.py::haversine_m`. Justification: trips span many
  latitudes and some are long; a single local equirectangular projection per
  trip is also provided (`project_local_xy`) for plotting/2-D geometry where
  a few-km extent makes the distortion negligible.
- Bearings use the standard initial-bearing formula (`bearing_deg`); heading
  changes use wrapped angular differences (`angular_diff_deg`, [-180,180)).
- Folium note: files are (lon, lat) but Folium wants (lat, lon) — handled
  explicitly in the visualisation layer (Step 3).

## Sampling / timing findings (training, 80 trips)

- Pooled share of steps exactly 1 s apart: **96.50%** (208 337 steps total,
  7 289 non-1 s) — consistent with the "~97%" expectation.
- 20/80 trips have *every* step == 1 s (`frac_dt_ne_1s == 0`).
- Timestamps are relative seconds, but `t_start` varies (1..693 s); they do
  NOT all begin at 0/1. Only differences matter.
- Irregular trips (verified, not assumed). Two distinct pathologies:
  - **Multi-second / irregular sampling** (high `frac_dt_ne_1s`):
    00000022 (47%), 00000004 (46%), 00000090 (42%), 00000039 (42%),
    00000046 (38%), 00000025 (31%), 00000081 (28%), 00000078 (24%),
    00000098 (23%), 00000026 (23%).
  - **Large temporal gaps** (max_gap_s > 60 s): 22 trips; worst are
    00000090 (153 064 s ≈ 42.5 h gap — likely a multi-day trace or long
    parked period), 00000078 (52 810 s), 00000069 (2 658 s),
    00000021 (2 161 s), 00000011 (909 s).
  - Note: 00000098 was flagged in earlier inspection but is only moderately
    irregular (23% non-1 s steps, max gap 50 s).
- Implication for features: `dt` is near-constant for ~75% of trips but
  heavy-tailed overall; speed computed as dist/dt will explode across gaps
  unless handled (windowed speed and gap indicator needed — Step 4).

## Data-quality validation results

- All 80 training trips pass validators: no NaN/inf, timestamps strictly
  increasing, positive dt, coordinates within valid WGS84 ranges, all arc
  endpoints in node range, all route arc ids in range.
- No problems found after fixing the (initially wrong) regional bounding-box
  check — see "worldwide trips" note above.

## Route / GPS consistency (Step 2)

- GPS -> nearest network node (cKDTree, projected metres):
  trip 00000057 median 16.3 m / p95 62.3 m; 00000018 median 10.2 m / p95 66.5 m;
  irregular 00000004 median 23.4 m / p95 138.3 m. Consistent with consumer-GPS
  noise; no evidence GPS is systematically off-network.
- GPS -> nearest *route* node median ~15-28 m (route nodes are sparser than
  all nodes, so this overestimates true route distance).
- Figures: results/figures/network_000000{57,18,04}.png

## Feature tables received (Step 4 inputs)

- `cleaned_gps_features.csv` (Person 3): 227,577 rows = 80 train + 10
  student_test trips; point-level features (time_diff_s, dist_m, speed,
  heading, flags, segment_id). No NaN/inf, no duplicate (trip_id, obs_index).
- `candidate_features_sample.csv`: SAMPLE of 3 training trips 
(00000001, 00000003, 00000004), 44,168 rows, ~13.8 candidates/obs,
  is_reference balance 7.2% positive. 
- `data_preprocessing.ipynb`: Person 3's pipeline doc. Claims all data is
  "Hessen, Germany" -- CONTRADICTED by Step 0 audit (trips worldwide).
  Their candidate projection assumed fixed lat0=50 degN; metric distances
  are inaccurate for trips far from 50 degN (e.g. trip 00000098 at -29 degS).
  Flagged to team; candidate features should be recomputed with a
  per-trip local projection.
- `feature_documentation.csv` HAS NOW been delivered (14 rows). It documents
  ONLY the point-level features of cleaned_gps_features.csv plus two
  "Planned" candidate-dependent rows. It does NOT document any column of
  candidate_features_sample.csv, does NOT mention projection/lat0, and does
  NOT list flag_zero_time, trip_id, partition or obs_index.
- Verified: candidate obs_idx == cleaned obs_index; every observation has
  >=1 candidate; is_reference arcs are a subset of .route arcs (no
  fabricated labels), but .route has more unique arcs than is_reference
  marks (e.g. trip1: 525 route vs 458 ref) -- expected, since a route arc
  only gets labelled where a fix falls within the 50 m candidate radius.

## Step 4 feature-analysis findings (training only)

Point-level (cleaned_gps_features.csv, 208,417 train rows, 80 trips):
- No NaN/inf; timestamps strictly increasing; all validators pass.
- time_diff_s: 96.5% exactly 1.0 s (201,048 rows); heavy tail to 153,064 s.
  Low-variance for most trips -> gap FLAGS carry more information than dt.
- speed_kmh by dt bin (median / p95): ~1s: 40.5/120.7; ~2s: 25.9/77.5;
  2-5s: 18.7/80.6; 5-60s: 13.8/64.9; >60s: 0.2/0.7 km/h -> single-step
  speed is meaningless across gaps; windowed speed needed.
- speed_ms and speed_kmh are exact duplicates (rho=1.0, VIF ~1e6): keep one.
- dist_m~speed rho=0.97. heading_deg in [0,360); first obs copies obs-1
  heading (100% match) -> no NaN but a mild forward-fill convention.
- flag_zero_time: 0 occurrences in train (dead feature candidate).
- flag_gap_gt_5s: 1,216 rows; flag_extreme_gap: 69 rows (defines segment_id).

Candidate-level (candidate_features_sample.csv, 44,168 rows, trips 1/3/4 ONLY):
- 13.8 candidates/obs (2-80); EVERY observation has >=1 candidate AND >=1
  is_reference=1 candidate. No NaN/inf; no duplicate (trip,obs,cand).
- is_reference balance 7.2% positive. proj_ratio within [0,1];
  dist_to_candidate_arc_m capped at 50.0 m; heading_delta_deg in [0,180]
  (absolute difference, already wrap-safe).
- Spearman: only perfect pair is gps_speed_ms~gps_speed_kmh (duplicates).
- VIF all ~1.0-12 except the speed duplicate pair.
- GroupKFold-by-trip Macro F1 (HistGB, 3 folds): base 0.7638;
  +lon/lat 0.7229 (delta -0.041, NO memorisation gain on this sample);
  +candidate_arc_id 0.7302 (delta -0.034, ids useless).
- Out-of-fold permutation importance (F1 drop): proj_ratio 0.177,
  dist_to_candidate_arc_m 0.145, heading_delta_deg 0.095, arc_length_m 0.038,
  gps_dist_m 0.018, rest ~0.
- Family ablation: drop candidate_geometry -> F1 0.478 (-0.286, dominant);
  drop transition_heading -> 0.693 (-0.071); drop gps_kinematics -> 0.720
  (-0.044); drop anomaly_flags -> 0.764 (0.000, redundant w/ other features).
- CAVEAT: only 3 trips -> CV is indicative; rerun when full candidate
  table arrives. lon/lat were joined from cleaned_gps_features for the
  memorisation test only; they are NOT in the candidate table itself.

Leakage assessment: is_reference is the label (never a feature);
connected_from_prev/heading_delta_deg use only t-1 context (causal, OK for
HMM); no t+1 features present; gps_* are per-obs (duplicated across that
obs's candidates) -- legitimate but not candidate-specific.

Artifacts: results/figures/step4_*.png (7 figs),
results/feature_selection_table.csv, notebooks/step4_feature_analysis.py.


## References

- Audit script: `notebooks/step0_timing_audit.py`
- Stats tables: `results/train_track_stats.csv`, `results/train_network_shapes.csv`
- Loader: `src/data/loading.py`

## AI assistance log

See `results/ai_assistance_log.md`.

---
*Last updated: Steps 1-4 complete + feature_documentation.csv review.*

## Evaluator frontend (Person 4, Phase 2)

- Stack: static HTML + vanilla JS + Leaflet (CDN). Chosen because Laptop 2
  needs only a browser — no Node/build step, no Python env. Leaflet gives
  toggleable layers + legend, mirroring src/visualization/folium_map.py
  ideas (which are server-side Python and cannot run in the browser).
- ui/config.js resolves the API base URL at runtime: ?api= query param >
  localStorage > config default. Never hardcoded to localhost, since
  Laptop 2 calls Laptop 1 over the LAN.
- Provisional API contract documented in ui/README.md and ui/app.js:
  GET /api/evaluator/trips + POST /api/evaluator/trips/{id}/predict.
  Backend team's evaluator routes were still in progress at build time;
  all calls are isolated in app.js normalisers so a schema change is a
  one-file fix.
- Ground-truth .route labels are never fetched or rendered; the UI shows
  only GPS observations and whatever route the API returns.
- Mock mode (?mock=1) loads ui/mocks/dev_mock.js which intercepts only
  the two evaluator endpoints and shows a red MOCK DATA banner; without
  the flag the module is never loaded, so real responses are the only
  ones that can appear in the shipped UI.
- evaluator_test/ (raw evaluator partition) and all raw dataset file
  types added to .gitignore — restricted data must never be committed.
