import time
import os
import glob
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss

print("="*85)
print("STAGE 1: GENERATING ENRICHED CANDIDATE FEATURES WITH ROBUST GROUND-TRUTH ALIGNMENT")
print("="*85)

os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('graphs', exist_ok=True)
os.makedirs('PathSnap_ML_Hackathon_Team-11/models', exist_ok=True)
os.makedirs('PathSnap_ML_Hackathon_Team-11/results', exist_ok=True)
os.makedirs('PathSnap_ML_Hackathon_Team-11/graphs', exist_ok=True)

ADVANCED_FEATURE_COLS = [
    # Baseline 11 features
    'dist_to_candidate_arc_m', 'proj_ratio', 'arc_length_m', 'arc_heading_deg',
    'heading_delta_deg', 'connected_from_prev', 'gps_speed_kmh', 'gps_dist_m',
    'gps_time_diff_s', 'flag_speed_anomaly', 'flag_gap_gt_5s',
    # Advanced 6 ranking & spatial features
    'cos_heading_diff',
    'dist_rank',
    'dist_ratio_to_min',
    'dist_diff_from_min',
    'dist_to_u_m',
    'dist_to_v_m'
]

def build_enriched_candidates(partition='train', radius_m=50.0):
    all_rows = []
    base_dir = f'docs/Student folder/{partition}'
    trip_dirs = sorted([d for d in glob.glob(os.path.join(base_dir, '*')) if os.path.isdir(d)])
    
    t0 = time.time()
    for td in trip_dirs:
        trip_id = os.path.basename(td)
        track_path = f'{td}/{trip_id}.track'
        nodes_path = f'{td}/{trip_id}.nodes'
        arcs_path = f'{td}/{trip_id}.arcs'
        route_path = f'{td}/{trip_id}.route'
        
        if not (os.path.isfile(track_path) and os.path.isfile(nodes_path) and os.path.isfile(arcs_path)):
            continue
            
        track_df = pd.read_csv(track_path, sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
        nodes_df = pd.read_csv(nodes_path, sep=r'\s+', header=None, names=['lon', 'lat'])
        arcs_df = pd.read_csv(arcs_path, sep=r'\s+', header=None, names=['u', 'v'])
        route_df = pd.read_csv(route_path, header=None, names=['arc_id']) if os.path.isfile(route_path) else None
        
        n_obs = len(track_df)
        lat0 = np.mean(nodes_df['lat'])
        m_per_deg_lat = 111139.0
        m_per_deg_lon = 111139.0 * np.cos(np.radians(lat0))
        
        node_x = nodes_df['lon'].values * m_per_deg_lon
        node_y = nodes_df['lat'].values * m_per_deg_lat
        
        u_idx, v_idx = arcs_df['u'].values, arcs_df['v'].values
        ax1, ay1 = node_x[u_idx], node_y[u_idx]
        ax2, ay2 = node_x[v_idx], node_y[v_idx]
        adx, ady = ax2 - ax1, ay2 - ay1
        arc_lengths = np.hypot(adx, ady)
        arc_headings = (np.degrees(np.arctan2(adx, ady)) + 360.0) % 360.0
        
        # Spatial Grid Index
        cell_size = float(radius_m)
        grid = {}
        min_x, max_x = np.minimum(ax1, ax2), np.maximum(ax1, ax2)
        min_y, max_y = np.minimum(ay1, ay2), np.maximum(ay1, ay2)
        for aid in range(len(arcs_df)):
            cx1, cx2 = int(min_x[aid]//cell_size), int(max_x[aid]//cell_size)
            cy1, cy2 = int(min_y[aid]//cell_size), int(max_y[aid]//cell_size)
            for cx in range(cx1, cx2+1):
                for cy in range(cy1, cy2+1):
                    grid.setdefault((cx, cy), []).append(aid)
                    
        tx = track_df['lon'].values * m_per_deg_lon
        ty = track_df['lat'].values * m_per_deg_lat
        ttime = track_df['time'].values
        dt = np.zeros(n_obs)
        dt[1:] = ttime[1:] - ttime[:-1]
        pdx, pdy = np.zeros(n_obs), np.zeros(n_obs)
        pdx[1:] = tx[1:] - tx[:-1]
        pdy[1:] = ty[1:] - ty[:-1]
        dist_m = np.hypot(pdx, pdy)
        with np.errstate(divide='ignore', invalid='ignore'):
            speed_ms = np.where(dt > 0, dist_m / dt, 0.0)
        gps_headings = (np.degrees(np.arctan2(pdx, pdy)) + 360.0) % 360.0
        if n_obs > 1: gps_headings[0] = gps_headings[1]
        
        # Robust Adaptive Ground Truth Alignment
        ref_arcs = []
        if route_df is not None and len(route_df) > 0:
            ref_route_arcs = route_df['arc_id'].values
            R = len(ref_route_arcs)
            ru, rv = arcs_df.iloc[ref_route_arcs]['u'].values, arcs_df.iloc[ref_route_arcs]['v'].values
            rx1, ry1 = node_x[ru], node_y[ru]
            rx2, ry2 = node_x[rv], node_y[rv]
            rdx, rdy = rx2 - rx1, ry2 - ry1
            rL2 = rdx*rdx + rdy*rdy
            
            cur_r = 0
            for i in range(n_obs):
                px, py = tx[i], ty[i]
                w_start = max(0, cur_r - 20)
                w_end = min(R, cur_r + 60)
                best_d, best_idx = float('inf'), cur_r
                for r_i in range(w_start, w_end):
                    dx, dy = rdx[r_i], rdy[r_i]
                    L2 = rL2[r_i]
                    t_val = 0.0 if L2==0 else max(0.0, min(1.0, ((px - rx1[r_i])*dx + (py - ry1[r_i])*dy)/L2))
                    d = np.hypot(px - (rx1[r_i] + t_val*dx), py - (ry1[r_i] + t_val*dy))
                    if d < best_d:
                        best_d, best_idx = d, r_i
                # Fallback to full route if window distance is excessively large
                if best_d > 45.0:
                    for r_i in range(R):
                        dx, dy = rdx[r_i], rdy[r_i]
                        L2 = rL2[r_i]
                        t_val = 0.0 if L2==0 else max(0.0, min(1.0, ((px - rx1[r_i])*dx + (py - ry1[r_i])*dy)/L2))
                        d = np.hypot(px - (rx1[r_i] + t_val*dx), py - (ry1[r_i] + t_val*dy))
                        if d < best_d:
                            best_d, best_idx = d, r_i
                cur_r = best_idx
                ref_arcs.append(ref_route_arcs[best_idx])
        else:
            ref_arcs = [-1] * n_obs
            
        # Candidate Extraction + Ranking Features
        prev_cand_arcs = set()
        prev_cand_nodes_v = set()
        
        for i in range(n_obs):
            px, py = tx[i], ty[i]
            cx1, cx2 = int((px - radius_m)//cell_size), int((px + radius_m)//cell_size)
            cy1, cy2 = int((py - radius_m)//cell_size), int((py + radius_m)//cell_size)
            cands = set()
            for cx in range(cx1, cx2+1):
                for cy in range(cy1, cy2+1):
                    if (cx, cy) in grid: cands.update(grid[(cx, cy)])
                    
            raw_cands = []
            for aid in cands:
                x1_val, y1_val = ax1[aid], ay1[aid]
                x2_val, y2_val = ax2[aid], ay2[aid]
                dx, dy = x2_val - x1_val, y2_val - y1_val
                L2 = dx*dx + dy*dy
                if L2 == 0:
                    t_val = 0.0
                    d_val = np.hypot(px - x1_val, py - y1_val)
                else:
                    t_val = max(0.0, min(1.0, ((px - x1_val)*dx + (py - y1_val)*dy) / L2))
                    d_val = np.hypot(px - (x1_val + t_val*dx), py - (y1_val + t_val*dy))
                if d_val <= radius_m:
                    h_diff = abs(gps_headings[i] - arc_headings[aid]) % 360.0
                    if h_diff > 180.0: h_diff = 360.0 - h_diff
                    is_connected = 1 if (aid in prev_cand_arcs or u_idx[aid] in prev_cand_nodes_v or i == 0) else 0
                    is_ref = 1 if (aid == ref_arcs[i]) else 0
                    
                    # Advanced distance to endpoints u and v
                    d_u = np.hypot(px - x1_val, py - y1_val)
                    d_v = np.hypot(px - x2_val, py - y2_val)
                    cos_h = np.cos(np.radians(h_diff))
                    
                    raw_cands.append({
                        'trip_id': trip_id,
                        'obs_idx': i,
                        'candidate_arc_id': aid,
                        'is_reference': is_ref,
                        'partition': partition,
                        'dist_to_candidate_arc_m': d_val,
                        'proj_ratio': t_val,
                        'arc_length_m': arc_lengths[aid],
                        'arc_heading_deg': arc_headings[aid],
                        'heading_delta_deg': h_diff,
                        'connected_from_prev': is_connected,
                        'gps_speed_kmh': speed_ms[i] * 3.6,
                        'gps_time_diff_s': dt[i],
                        'gps_dist_m': dist_m[i],
                        'flag_speed_anomaly': 1 if speed_ms[i] > 50.0 else 0,
                        'flag_gap_gt_5s': 1 if dt[i] > 5.0 else 0,
                        'cos_heading_diff': cos_h,
                        'dist_to_u_m': d_u,
                        'dist_to_v_m': d_v
                    })
                    
            if len(raw_cands) > 0:
                # Add relative ranking features within this observation
                raw_cands.sort(key=lambda c: c['dist_to_candidate_arc_m'])
                min_dist = raw_cands[0]['dist_to_candidate_arc_m']
                for rank_idx, c in enumerate(raw_cands):
                    c['dist_rank'] = rank_idx + 1
                    c['dist_ratio_to_min'] = c['dist_to_candidate_arc_m'] / (min_dist + 0.1)
                    c['dist_diff_from_min'] = c['dist_to_candidate_arc_m'] - min_dist
                    
            prev_cand_arcs = set(c['candidate_arc_id'] for c in raw_cands)
            prev_cand_nodes_v = set(v_idx[c['candidate_arc_id']] for c in raw_cands)
            all_rows.extend(raw_cands)
            
    print(f"Generated {len(all_rows):,} enriched candidates for {partition} in {time.time()-t0:.2f}s")
    return pd.DataFrame(all_rows)

TRAIN_V2_CACHE = 'data/candidates_train_80_trips_v2.pkl'
TEST_V2_CACHE = 'data/candidates_test_10_trips_v2.pkl'

if os.path.exists(TRAIN_V2_CACHE):
    print(f"Loading cached enriched train data from {TRAIN_V2_CACHE}...")
    with open(TRAIN_V2_CACHE, 'rb') as f:
        train_df = pickle.load(f)
else:
    print("Building enriched training candidates (80 trips)...")
    train_df = build_enriched_candidates('train', radius_m=50.0)
    with open(TRAIN_V2_CACHE, 'wb') as f:
        pickle.dump(train_df, f, protocol=pickle.HIGHEST_PROTOCOL)

if os.path.exists(TEST_V2_CACHE):
    print(f"Loading cached enriched test data from {TEST_V2_CACHE}...")
    with open(TEST_V2_CACHE, 'rb') as f:
        test_df = pickle.load(f)
else:
    print("Building enriched test candidates (10 trips)...")
    test_df = build_enriched_candidates('student_test', radius_m=50.0)
    with open(TEST_V2_CACHE, 'wb') as f:
        pickle.dump(test_df, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"\nTraining Set Shape: {train_df.shape} | Positive reference labels: {train_df['is_reference'].sum():,} ({train_df['is_reference'].mean()*100:.2f}%)")
print(f"Test Set Shape:     {test_df.shape} | Positive reference labels: {test_df['is_reference'].sum():,} ({test_df['is_reference'].mean()*100:.2f}%)")

# -----------------------------------------------------------------------------
# 2. TRAIN ADVANCED MODELS: OPTUNA-TUNED RF & HISTGRADIENTBOOSTING
# -----------------------------------------------------------------------------
print("\n" + "="*85)
print("STAGE 2: TRAINING ADVANCED MODELS ON ENRICHED FEATURES")
print("="*85)

X_train = train_df[ADVANCED_FEATURE_COLS].values
y_train = train_df['is_reference'].values

# Model 1: Optuna Tuned Random Forest with Enriched Features
print("Training Advanced Random Forest (100 trees, depth 14, leaf 8, subsample 0.55)...")
t0 = time.time()
rf_advanced = RandomForestClassifier(
    n_estimators=100,
    max_depth=14,
    min_samples_leaf=8,
    min_samples_split=6,
    max_features='sqrt',
    max_samples=0.55,
    criterion='gini',
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
rf_advanced.fit(X_train, y_train)
rf_train_time = time.time() - t0
print(f"Random Forest Training Completed in {rf_train_time:.2f}s")
joblib.dump(rf_advanced, 'models/best_model_advanced_rf.joblib')
joblib.dump(rf_advanced, 'PathSnap_ML_Hackathon_Team-11/models/best_model_advanced_rf.joblib')

# Model 2: HistGradientBoosting
print("\nTraining Advanced HistGradientBoosting (120 iterations, lr 0.08)...")
t0 = time.time()
hgb_advanced = HistGradientBoostingClassifier(
    max_iter=120,
    learning_rate=0.08,
    max_depth=12,
    min_samples_leaf=15,
    class_weight='balanced',
    random_state=42
)
hgb_advanced.fit(X_train, y_train)
hgb_train_time = time.time() - t0
print(f"HistGradientBoosting Training Completed in {hgb_train_time:.2f}s")
joblib.dump(hgb_advanced, 'models/best_model_advanced_hgb.joblib')
joblib.dump(hgb_advanced, 'PathSnap_ML_Hackathon_Team-11/models/best_model_advanced_hgb.joblib')

# Feature Importances from Random Forest
feat_imp = pd.DataFrame({
    'feature': ADVANCED_FEATURE_COLS,
    'importance': rf_advanced.feature_importances_
}).sort_values(by='importance', ascending=False).reset_index(drop=True)

print("\n--- TOP FEATURE IMPORTANCES (RANDOM FOREST) ---")
print(feat_imp.to_string(index=False))

# -----------------------------------------------------------------------------
# 3. BENCHMARK WITH HMM VITERBI ON ALL 10 STUDENT-TEST TRIPS
# -----------------------------------------------------------------------------
print("\n" + "="*85)
print("STAGE 3: FULL HMM VITERBI EVALUATION ON ALL 10 STUDENT-TEST TRIPS")
print("="*85)

test_trips = sorted(test_df['trip_id'].unique())

def evaluate_trip_models(trip_id):
    trip_cands = test_df[test_df['trip_id'] == trip_id].copy()
    if len(trip_cands) == 0:
        return None
        
    td = f'docs/Student folder/student_test/{trip_id}'
    nodes_df = pd.read_csv(f'{td}/{trip_id}.nodes', sep=r'\s+', header=None, names=['lon', 'lat'])
    arcs_df = pd.read_csv(f'{td}/{trip_id}.arcs', sep=r'\s+', header=None, names=['u', 'v'])
    route_df = pd.read_csv(f'{td}/{trip_id}.route', header=None, names=['arc_id'])
    track_df = pd.read_csv(f'{td}/{trip_id}.track', sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
    
    # Model predictions
    probs_rf = rf_advanced.predict_proba(trip_cands[ADVANCED_FEATURE_COLS].values)[:, 1]
    probs_hgb = hgb_advanced.predict_proba(trip_cands[ADVANCED_FEATURE_COLS].values)[:, 1]
    # Blended probability
    probs_blend = 0.60 * probs_rf + 0.40 * probs_hgb
    
    trip_cands['prob_rf'] = probs_rf
    trip_cands['prob_blend'] = probs_blend
    
    obs_groups = [group for _, group in trip_cands.groupby('obs_idx')]
    T = len(obs_groups)
    
    # Ground Truth Reference Arcs
    ref_route_arcs = route_df['arc_id'].values
    R = len(ref_route_arcs)
    lat0 = np.mean(nodes_df['lat'])
    m_per_deg_lat = 111139.0
    m_per_deg_lon = 111139.0 * np.cos(np.radians(lat0))
    node_x = nodes_df['lon'].values * m_per_deg_lon
    node_y = nodes_df['lat'].values * m_per_deg_lat
    tx = track_df['lon'].values * m_per_deg_lon
    ty = track_df['lat'].values * m_per_deg_lat
    ru, rv = arcs_df.iloc[ref_route_arcs]['u'].values, arcs_df.iloc[ref_route_arcs]['v'].values
    rx1, ry1 = node_x[ru], node_y[ru]
    rx2, ry2 = node_x[rv], node_y[rv]
    rdx, rdy = rx2 - rx1, ry2 - ry1
    rL2 = rdx*rdx + rdy*rdy
    
    cur_r = 0
    true_obs_arcs = []
    for i in range(T):
        px, py = tx[i], ty[i]
        w_start = max(0, cur_r - 20)
        w_end = min(R, cur_r + 60)
        best_d, best_idx = float('inf'), cur_r
        for r_i in range(w_start, w_end):
            dx, dy = rdx[r_i], rdy[r_i]
            L2 = rL2[r_i]
            t_val = 0.0 if L2==0 else max(0.0, min(1.0, ((px - rx1[r_i])*dx + (py - ry1[r_i])*dy)/L2))
            d = np.hypot(px - (rx1[r_i] + t_val*dx), py - (ry1[r_i] + t_val*dy))
            if d < best_d:
                best_d, best_idx = d, r_i
        if best_d > 45.0:
            for r_i in range(R):
                dx, dy = rdx[r_i], rdy[r_i]
                L2 = rL2[r_i]
                t_val = 0.0 if L2==0 else max(0.0, min(1.0, ((px - rx1[r_i])*dx + (py - ry1[r_i])*dy)/L2))
                d = np.hypot(px - (rx1[r_i] + t_val*dx), py - (ry1[r_i] + t_val*dy))
                if d < best_d:
                    best_d, best_idx = d, r_i
        cur_r = best_idx
        true_obs_arcs.append(ref_route_arcs[best_idx])
    true_obs_arcs = np.array(true_obs_arcs)
    
    # Viterbi DP Decoder
    def run_viterbi(prob_col):
        eps = 1e-6
        first_cands = obs_groups[0]['candidate_arc_id'].values
        first_probs = obs_groups[0][prob_col].values
        first_dists = obs_groups[0]['dist_to_candidate_arc_m'].values
        first_emiss = np.log(first_probs + eps) - 0.5 * (first_dists / 15.0)**2
        
        dp = {arc: (first_emiss[i], [arc]) for i, arc in enumerate(first_cands)}
        
        for t in range(1, T):
            curr_cands = obs_groups[t]['candidate_arc_id'].values
            curr_probs = obs_groups[t][prob_col].values
            curr_dists = obs_groups[t]['dist_to_candidate_arc_m'].values
            curr_emiss = np.log(curr_probs + eps) - 0.5 * (curr_dists / 15.0)**2
            
            new_dp = {}
            for c_idx, curr_arc in enumerate(curr_cands):
                e_prob = curr_emiss[c_idx]
                best_prev_score = -float('inf')
                best_path = None
                curr_u = arcs_df.iloc[curr_arc]['u']
                
                for prev_arc, (prev_score, path) in dp.items():
                    prev_v = arcs_df.iloc[prev_arc]['v']
                    if prev_arc == curr_arc:
                        trans_cost = 0.0
                    elif prev_v == curr_u:
                        trans_cost = -0.3
                    else:
                        trans_cost = -4.0
                    total_s = prev_score + trans_cost + e_prob
                    if total_s > best_prev_score:
                        best_prev_score = total_s
                        best_path = path + [curr_arc]
                if best_path is not None:
                    new_dp[curr_arc] = (best_prev_score, best_path)
            dp = new_dp if new_dp else {arc: (curr_emiss[i], [arc]) for i, arc in enumerate(curr_cands)}
            
        best_final_arc = max(dp.keys(), key=lambda k: dp[k][0])
        viterbi_route = np.array(dp[best_final_arc][1])
        
        # Classifier top1
        clf_top1 = np.array([g.loc[g[prob_col].idxmax(), 'candidate_arc_id'] for g in obs_groups])
        
        acc_clf = np.mean(clf_top1 == true_obs_arcs)
        acc_vit = np.mean(viterbi_route == true_obs_arcs)
        
        cont = 0
        for i in range(len(viterbi_route)-1):
            a1, a2 = viterbi_route[i], viterbi_route[i+1]
            if a1 == a2 or arcs_df.iloc[a1]['v'] == arcs_df.iloc[a2]['u']:
                cont += 1
        continuity = cont / (len(viterbi_route)-1) if len(viterbi_route) > 1 else 1.0
        
        return acc_clf, acc_vit, continuity
        
    clf_rf, vit_rf, cont_rf = run_viterbi('prob_rf')
    clf_bl, vit_bl, cont_bl = run_viterbi('prob_blend')
    
    return {
        'trip_id': trip_id,
        'n_obs': T,
        'rf_clf_acc': round(clf_rf * 100, 2),
        'rf_hmm_acc': round(vit_rf * 100, 2),
        'rf_continuity': round(cont_rf * 100, 2),
        'blended_clf_acc': round(clf_bl * 100, 2),
        'blended_hmm_acc': round(vit_bl * 100, 2),
        'blended_continuity': round(cont_bl * 100, 2)
    }

results = []
for tid in test_trips:
    res = evaluate_trip_models(tid)
    results.append(res)

eval_df = pd.DataFrame(results)

total_obs = eval_df['n_obs'].sum()
avg_rf_clf = np.average(eval_df['rf_clf_acc'], weights=eval_df['n_obs'])
avg_rf_hmm = np.average(eval_df['rf_hmm_acc'], weights=eval_df['n_obs'])
avg_rf_cont = np.average(eval_df['rf_continuity'], weights=eval_df['n_obs'])

avg_bl_clf = np.average(eval_df['blended_clf_acc'], weights=eval_df['n_obs'])
avg_bl_hmm = np.average(eval_df['blended_hmm_acc'], weights=eval_df['n_obs'])
avg_bl_cont = np.average(eval_df['blended_continuity'], weights=eval_df['n_obs'])

print("\n" + "="*85)
print("TRIP-BY-TRIP EVALUATION WITH ADVANCED RANKING FEATURES & ROBUST ALIGNMENT")
print("="*85)
print(eval_df[['trip_id', 'n_obs', 'rf_clf_acc', 'rf_hmm_acc', 'rf_continuity', 'blended_hmm_acc']].to_string(index=False))

print("-" * 85)
print(f"Advanced Random Forest Classifier:  {avg_rf_clf:.2f}%")
print(f"Advanced Random Forest + HMM:       {avg_rf_hmm:.2f}%  (Continuity: {avg_rf_cont:.2f}%)")
print(f"Blended Ensemble (RF+HGB) + HMM:    {avg_bl_hmm:.2f}%  (Continuity: {avg_bl_cont:.2f}%)")
print("="*85)

eval_df.to_csv('results/advanced_features_trip_breakdown.csv', index=False)
eval_df.to_csv('PathSnap_ML_Hackathon_Team-11/results/advanced_features_trip_breakdown.csv', index=False)

# -----------------------------------------------------------------------------
# 4. PLOT PERFORMANCE COMPARISON
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Feature Importances
top_feats = feat_imp.head(10).iloc[::-1]
ax1 = axes[0]
bars = ax1.barh(top_feats['feature'], top_feats['importance'] * 100, color='#2b5c8f', edgecolor='black', alpha=0.85)
ax1.set_xlabel('Relative Feature Importance (%)', fontsize=11, fontweight='bold')
ax1.set_title('Top 10 Feature Importances (Random Forest)', fontsize=12, fontweight='bold')
ax1.grid(axis='x', linestyle='--', alpha=0.5)
for bar in bars:
    xval = bar.get_width()
    ax1.text(xval + 0.5, bar.get_y() + bar.get_height()/2.0, f"{xval:.1f}%", va='center', fontsize=9, fontweight='bold')

# Plot 2: Trip-by-Trip Accuracy
ax2 = axes[1]
x = np.arange(len(eval_df))
width = 0.35
b1 = ax2.bar(x - width/2, eval_df['rf_clf_acc'], width, label=f'Classifier Top-1 ({avg_rf_clf:.2f}%)', color='#4682b4', alpha=0.85, edgecolor='black')
b2 = ax2.bar(x + width/2, eval_df['rf_hmm_acc'], width, label=f'HMM Viterbi ({avg_rf_hmm:.2f}%)', color='#2ca02c', alpha=0.85, edgecolor='black')

ax2.set_xticks(x)
ax2.set_xticklabels([f"Trip {tid}" for tid in eval_df['trip_id']], rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('Trajectory Accuracy (%)', fontsize=11, fontweight='bold')
ax2.set_title(f'Trip Accuracy with Advanced Features (Overall: {avg_rf_hmm:.2f}%)', fontsize=12, fontweight='bold')
ax2.legend(loc='lower left', frameon=True)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('graphs/advanced_features_accuracy_jump.png', dpi=150)
plt.savefig('PathSnap_ML_Hackathon_Team-11/graphs/advanced_features_accuracy_jump.png', dpi=150)
plt.close()

print("Saved plots to graphs/advanced_features_accuracy_jump.png")
print("Advanced feature training & evaluation successfully completed!")
