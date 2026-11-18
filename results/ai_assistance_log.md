# AI assistance log

One line per task: what was generated, for whom, and when.

| Date | Task | What AI generated | Used by |
|------|------|-------------------|---------|
| 2026-11-18 | Step 0: repo scaffolding | Folder structure (`src/data`, `src/visualization`, `src/features`, `notebooks`, `results/figures`, `results/maps`) and `__init__.py` files | Person 4 |
| 2026-11-18 | Step 0: data loader | `src/data/loading.py` — loaders for .track/.nodes/.arcs/.route, validators, haversine/bearing/angular-diff helpers | Person 4 |
| 2026-11-18 | Step 0: timing audit | `notebooks/step0_timing_audit.py` + `results/train_track_stats.csv`, `results/train_network_shapes.csv` (all 80 training trips) | Person 4 |
| 2026-11-18 | Step 0: decisions log | `DECISIONS.md` — layout findings, worldwide-trips note, irregular-trip list, open questions | Person 4 |

| 2026-11-18 | Step 1: trajectory plots | `notebooks/step1_traj_plots.py` + 10 PNGs in `results/figures/traj_*.png` (3 representative + 7 irregular training trips) | Person 4 |
| 2026-11-18 | Step 2: network plots | `notebooks/step2_network_plots.py` + 3 PNGs `network_*.png` (clipped arcs, reference route, GPS, distance stats) | Person 4 |
| 2026-11-18 | Step 3: Folium maps | `src/visualization/folium_map.py`, `notebooks/step3_folium_maps.py` + 3 HTML maps in `results/maps/` | Person 4 |
| 2026-11-18 | Step 4a: feature inspection | Inspection of `cleaned_gps_features.csv`, `candidate_features_sample.csv`, `data_preprocessing.ipynb` (Person 3 docs) | Person 4 |

| 2026-11-18 | Step 4b: feature analysis | `notebooks/step4_feature_analysis.py` + 7 figs `step4_*.png` + `results/feature_selection_table.csv` (distributions, Spearman, VIF, GroupKFold CV, permutation importance, family ablation) | Person 4 |

| 2026-11-18 | Doc cross-check + lat0 verification | Compared `feature_documentation.csv` vs both tables; empirically verified candidate-feature definitions (distance/proj_ratio/arc_length/headings/gps_* copies) and disproved the lat0=50 concern; updated `feature_selection_table.csv` with doc columns | Person 4 |

*Append one row per task as work proceeds.*
| 2026-11-18 | Evaluator frontend | `ui/` — index.html, app.js (API client + normalisers + Leaflet map/metrics), config.js (runtime API base URL), styles.css, mocks/dev_mock.js (dev-only), README.md with run instructions and provisional API contract | Person 4 |
| 2026-11-18 | Gitignore hardening | Added evaluator_test/, Student folder/, *.track/.nodes/.arcs/.route, cleaned_gps_features.csv, results/maps/*.html rules so restricted data cannot be committed | Person 4 |
