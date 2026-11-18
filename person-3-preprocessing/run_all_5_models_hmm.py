import time
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

print("="*85)
print("COMPARING ALL 5 MODELS: CLASSIFIER-ONLY vs. CLASSIFIER + HMM VITERBI")
print("="*85)

from train_full_80_trips import build_partition_candidates

feature_cols = [
    'dist_to_candidate_arc_m', 'proj_ratio', 'arc_length_m', 'arc_heading_deg',
    'heading_delta_deg', 'connected_from_prev', 'gps_speed_kmh', 'gps_dist_m',
    'gps_time_diff_s', 'flag_speed_anomaly', 'flag_gap_gt_5s'
]

print("1. Generating/Loading Training Candidates (80 trips)...")
train_df = build_partition_candidates('train', radius_m=50.0)
X_train = train_df[feature_cols].values
y_train = train_df['is_reference'].values

models = {
    "Logistic Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=12, min_samples_leaf=25, class_weight='balanced', random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=14, min_samples_leaf=15, class_weight='balanced', 
        max_samples=0.25, n_jobs=-1, random_state=42
    ),
    "K-Nearest Neighbors": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=15, weights='distance', n_jobs=-1))
    ]),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.08, class_weight='balanced', random_state=42
    )
}

trained_models = {}
for name, model in models.items():
    t0 = time.time()
    if name == "K-Nearest Neighbors":
        sample_idx = np.random.RandomState(42).choice(len(X_train), size=min(40000, len(X_train)), replace=False)
        model.fit(X_train[sample_idx], y_train[sample_idx])
    else:
        model.fit(X_train, y_train)
    print(f"  Fitted {name:<22} in {time.time()-t0:.2f}s")
    trained_models[name] = model

# 2. Pre-generate test trips data structure
test_dir = 'docs/Student folder/student_test'
test_trips = sorted([d for d in glob.glob(os.path.join(test_dir, '*')) if os.path.isdir(d)])

print("\n2. Processing test candidate structures for all 10 trips...")
test_data = []
for td in test_trips:
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
    
    cell_size = 50.0
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
    
    # Ground truth mapping
    r_arcs = route_df['arc_id'].values
    rx1, ry1 = node_x[arcs_df.iloc[r_arcs]['u'].values], node_y[arcs_df.iloc[r_arcs]['u'].values]
    rx2, ry2 = node_x[arcs_df.iloc[r_arcs]['v'].values], node_y[arcs_df.iloc[r_arcs]['v'].values]
    cur_r = 0
    ref_arcs = []
    for i in range(n_obs):
        px, py = tx[i], ty[i]
        best_d, best_idx = float('inf'), cur_r
        w_start, w_end = max(0, cur_r - 2), min(len(r_arcs), cur_r + 25)
        for r_i in range(w_start, w_end):
            dx, dy = rx2[r_i] - rx1[r_i], ry2[r_i] - ry1[r_i]
            L2 = dx*dx + dy*dy
            t = 0.0 if L2==0 else max(0.0, min(1.0, ((px - rx1[r_i])*dx + (py - ry1[r_i])*dy)/L2))
            d = np.hypot(px - (rx1[r_i] + t*dx), py - (ry1[r_i] + t*dy))
            if d < best_d:
                best_d, best_idx = d, r_i
        cur_r = best_idx
        ref_arcs.append(r_arcs[best_idx])
        
    out_edges = {}
    for aid in range(len(arcs_df)):
        u_node = u_idx[aid]
        if u_node not in out_edges:
            out_edges[u_node] = set()
        out_edges[u_node].add(aid)
        
    obs_candidates = []
    all_features = []
    cand_map = []
    prev_cand_arcs = set()
    prev_cand_nodes_v = set()
    
    for i in range(n_obs):
        px, py = tx[i], ty[i]
        cx1, cx2 = int((px - cell_size)//cell_size), int((px + cell_size)//cell_size)
        cy1, cy2 = int((py - cell_size)//cell_size), int((py + cell_size)//cell_size)
        cands = set()
        for cx in range(cx1, cx2+1):
            for cy in range(cy1, cy2+1):
                if (cx, cy) in grid: cands.update(grid[(cx, cy)])
        cur_cands = []
        for aid in cands:
            x1_val, y1_val = ax1[aid], ay1[aid]
            x2_val, y2_val = ax2[aid], ay2[aid]
            dx, dy = x2_val - x1_val, y2_val - y1_val
            L2 = dx*dx + dy*dy
            t_val = 0.0 if L2==0 else max(0.0, min(1.0, ((px - x1_val)*dx + (py - y1_val)*dy)/L2))
            d_val = np.hypot(px - (x1_val + t_val*dx), py - (y1_val + t_val*dy))
            if d_val <= cell_size:
                h_diff = abs(gps_headings[i] - arc_headings[aid]) % 360.0
                if h_diff > 180.0: h_diff = 360.0 - h_diff
                is_conn = 1 if (aid in prev_cand_arcs or u_idx[aid] in prev_cand_nodes_v or i == 0) else 0
                cur_cands.append({
                    'candidate_arc_id': aid,
                    'u': u_idx[aid],
                    'v': v_idx[aid],
                    'dist_to_candidate_arc_m': d_val,
                    'proj_ratio': t_val,
                    'arc_length_m': arc_lengths[aid],
                    'arc_heading_deg': arc_headings[aid],
                    'heading_delta_deg': h_diff,
                    'connected_from_prev': is_conn,
                    'gps_speed_kmh': speed_ms[i] * 3.6,
                    'gps_time_diff_s': dt[i],
                    'gps_dist_m': dist_m[i],
                    'flag_speed_anomaly': 1 if speed_ms[i] > 50.0 else 0,
                    'flag_gap_gt_5s': 1 if dt[i] > 5.0 else 0
                })
        prev_cand_arcs = set(c['candidate_arc_id'] for c in cur_cands)
        prev_cand_nodes_v = set(c['v'] for c in cur_cands)
        obs_candidates.append(cur_cands)
        for j, c in enumerate(cur_cands):
            all_features.append([c[k] for k in feature_cols])
            cand_map.append((i, j))
            
    test_data.append({
        'trip_id': trip_id,
        'n_obs': n_obs,
        'dt': dt,
        'v_idx': v_idx,
        'out_edges': out_edges,
        'ref_arcs': ref_arcs,
        'obs_candidates': obs_candidates,
        'all_features': np.array(all_features),
        'cand_map': cand_map
    })

print(f"Processed {len(test_data)} test trips successfully.")

# 3. Benchmark all 5 models with Viterbi
print("\n" + "="*85)
print(f"{'Model':<24} | {'Classifier Acc':<16} | {'HMM Viterbi Acc':<16} | {'Gain':<10} | {'Continuity':<10}")
print("-"*85)

summary_rows = []

for model_name, model in trained_models.items():
    trip_clf_accs = []
    trip_vit_accs = []
    trip_conts = []
    
    for t_item in test_data:
        n_obs = t_item['n_obs']
        dt = t_item['dt']
        v_idx = t_item['v_idx']
        out_edges = t_item['out_edges']
        ref_arcs = t_item['ref_arcs']
        
        # Deepcopy candidates
        cands_copy = [[dict(c) for c in c_list] for c_list in t_item['obs_candidates']]
        
        # Batch predict
        if len(t_item['all_features']) > 0:
            probs = model.predict_proba(t_item['all_features'])[:, 1]
            for idx, (i_obs, j_cand) in enumerate(t_item['cand_map']):
                cands_copy[i_obs][j_cand]['prob'] = float(probs[idx])
                
        # 1. Classifier-Only
        clf_preds = []
        for i_obs in range(n_obs):
            clist = cands_copy[i_obs]
            if len(clist) == 0:
                clf_preds.append(-1)
            else:
                best_c = max(clist, key=lambda x: x.get('prob', 0.0))
                clf_preds.append(best_c['candidate_arc_id'])
        clf_acc = np.mean([1 if clf_preds[i] == ref_arcs[i] else 0 for i in range(n_obs)])
        trip_clf_accs.append(clf_acc)
        
        # 2. HMM Viterbi
        V_prev = {}
        BP = [{}]
        if len(cands_copy) > 0 and len(cands_copy[0]) > 0:
            for j, c in enumerate(cands_copy[0]):
                V_prev[j] = np.log(max(c.get('prob', 1e-6), 1e-6))
                
        for t in range(1, n_obs):
            c_prev = cands_copy[t-1]
            c_curr = cands_copy[t]
            V_curr = {}
            bp_curr = {}
            is_gap = dt[t] > 60.0
            
            for j, c_j in enumerate(c_curr):
                best_val = -float('inf')
                best_prev_idx = 0
                p_emit = np.log(max(c_j.get('prob', 1e-6), 1e-6))
                aid_j = c_j['candidate_arc_id']
                u_j = c_j['u']
                
                for i_prev, c_i in enumerate(c_prev):
                    aid_i = c_i['candidate_arc_id']
                    v_i = c_i['v']
                    
                    if is_gap:
                        trans_prob = 1.0 / max(1, len(c_prev))
                    elif aid_i == aid_j:
                        trans_prob = 0.85
                    elif aid_j in out_edges.get(v_i, set()):
                        trans_prob = 0.80
                    elif v_i == u_j:
                        trans_prob = 0.75
                    else:
                        two_hop = False
                        for mid_arc in out_edges.get(v_i, set()):
                            if aid_j in out_edges.get(v_idx[mid_arc], set()):
                                two_hop = True
                                break
                        trans_prob = 0.20 if two_hop else 1e-4
                        
                    val = V_prev.get(i_prev, -1e9) + np.log(trans_prob)
                    if val > best_val:
                        best_val = val
                        best_prev_idx = i_prev
                V_curr[j] = best_val + p_emit
                bp_curr[j] = best_prev_idx
            V_prev = V_curr
            BP.append(bp_curr)
            
        viterbi_preds = [-1] * n_obs
        if len(V_prev) > 0:
            best_last_idx = max(V_prev, key=V_prev.get)
            viterbi_preds[-1] = cands_copy[-1][best_last_idx]['candidate_arc_id']
            cur_idx = best_last_idx
            for t in range(n_obs - 2, -1, -1):
                cur_idx = BP[t+1].get(cur_idx, 0)
                if len(cands_copy[t]) > 0 and cur_idx < len(cands_copy[t]):
                    viterbi_preds[t] = cands_copy[t][cur_idx]['candidate_arc_id']
                    
        vit_acc = np.mean([1 if viterbi_preds[i] == ref_arcs[i] else 0 for i in range(n_obs)])
        trip_vit_accs.append(vit_acc)
        
        # Continuity
        valid_trans = 0
        for t in range(n_obs - 1):
            a1_v, a2_v = viterbi_preds[t], viterbi_preds[t+1]
            if a1_v != -1 and a2_v != -1:
                if a1_v == a2_v or a2_v in out_edges.get(v_idx[a1_v], set()):
                    valid_trans += 1
        cont = valid_trans / max(1, n_obs - 1)
        trip_conts.append(cont)
        
    avg_clf = np.mean(trip_clf_accs) * 100
    avg_vit = np.mean(trip_vit_accs) * 100
    avg_gain = avg_vit - avg_clf
    avg_cont = np.mean(trip_conts) * 100
    
    summary_rows.append({
        'Model': model_name,
        'Classifier Only Acc (%)': round(avg_clf, 2),
        'HMM Viterbi Acc (%)': round(avg_vit, 2),
        'Accuracy Gain (%)': round(avg_gain, 2),
        'Route Continuity (%)': round(avg_cont, 2)
    })
    print(f"{model_name:<24} | {avg_clf:6.2f}%          | {avg_vit:6.2f}%           | +{avg_gain:5.2f}%    | {avg_cont:5.2f}%")

print("="*85)
df_res = pd.DataFrame(summary_rows)

df_res.to_csv('results/all_5_models_hmm_comparison.csv', index=False)
df_res.to_csv('PathSnap_ML_Hackathon_Team-11/results/all_5_models_hmm_comparison.csv', index=False)

# Bar chart
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(df_res))
width = 0.35

rects1 = ax.bar(x - width/2, df_res['Classifier Only Acc (%)'], width, label='Classifier-Only (Pointwise)', color='#4285F4')
rects2 = ax.bar(x + width/2, df_res['HMM Viterbi Acc (%)'], width, label='Classifier + HMM Viterbi', color='#34A853')

ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('PathSnap: Standalone Classifier vs. Classifier + HMM Viterbi for All 5 Models (10 Test Trips)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df_res['Model'], rotation=15)
ax.set_ylim(40, 95)
ax.legend(fontsize=11)
ax.grid(True, linestyle='--', alpha=0.5)

for bar in rects1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}%", ha='center', va='bottom', fontsize=10)
for bar in rects2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('graphs/all_5_models_hmm_comparison.png', dpi=150)
plt.savefig('PathSnap_ML_Hackathon_Team-11/graphs/all_5_models_hmm_comparison.png', dpi=150)
plt.close()

print("\nDone! Saved results/all_5_models_hmm_comparison.csv and graphs/all_5_models_hmm_comparison.png")
