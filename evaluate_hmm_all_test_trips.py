import time
import os
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("="*80)
print("EVALUATING CLASSIFIER vs. CLASSIFIER + HMM VITERBI ON ALL 10 STUDENT-TEST TRIPS")
print("="*80)

# Load the trained Random Forest model
rf_model = joblib.load('models/best_model_full.joblib')

feature_cols = [
    'dist_to_candidate_arc_m', 'proj_ratio', 'arc_length_m', 'arc_heading_deg',
    'heading_delta_deg', 'connected_from_prev', 'gps_speed_kmh', 'gps_dist_m',
    'gps_time_diff_s', 'flag_speed_anomaly', 'flag_gap_gt_5s'
]

test_dir = 'docs/Student folder/student_test'
test_trips = sorted([d for d in glob.glob(os.path.join(test_dir, '*')) if os.path.isdir(d)])

results = []

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
    
    # 2D Grid
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
        
    # Adjacency lookup
    out_edges = {}
    for aid in range(len(arcs_df)):
        u_node = u_idx[aid]
        if u_node not in out_edges:
            out_edges[u_node] = set()
        out_edges[u_node].add(aid)
        
    # Generate candidate sets per observation fix
    obs_candidates = []
    all_cand_features = []
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
            if L2 == 0:
                t_val = 0.0
                d_val = np.hypot(px - x1_val, py - y1_val)
            else:
                t_val = max(0.0, min(1.0, ((px - x1_val)*dx + (py - y1_val)*dy) / L2))
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
            all_cand_features.append([c[k] for k in feature_cols])
            cand_map.append((i, j))
            
    # Batch predict probabilities
    if len(all_cand_features) > 0:
        probs = rf_model.predict_proba(np.array(all_cand_features))[:, 1]
        for idx, (i_obs, j_cand) in enumerate(cand_map):
            obs_candidates[i_obs][j_cand]['prob'] = float(probs[idx])
            
    # Baseline: Classifier-Only
    clf_only_preds = []
    for i, c_list in enumerate(obs_candidates):
        if len(c_list) == 0:
            clf_only_preds.append(-1)
        else:
            best_c = max(c_list, key=lambda x: x.get('prob', 0.0))
            clf_only_preds.append(best_c['candidate_arc_id'])
    clf_acc = np.mean([1 if clf_only_preds[i] == ref_arcs[i] else 0 for i in range(n_obs)])
    
    # HMM Viterbi Decoding
    t_dec_start = time.time()
    V_prev = {}
    BP = [{}]
    if len(obs_candidates) > 0 and len(obs_candidates[0]) > 0:
        for j, c in enumerate(obs_candidates[0]):
            V_prev[j] = np.log(max(c.get('prob', 1e-6), 1e-6))
            
    for t in range(1, n_obs):
        c_prev = obs_candidates[t-1]
        c_curr = obs_candidates[t]
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
        viterbi_preds[-1] = obs_candidates[-1][best_last_idx]['candidate_arc_id']
        cur_idx = best_last_idx
        for t in range(n_obs - 2, -1, -1):
            cur_idx = BP[t+1].get(cur_idx, 0)
            if len(obs_candidates[t]) > 0 and cur_idx < len(obs_candidates[t]):
                viterbi_preds[t] = obs_candidates[t][cur_idx]['candidate_arc_id']
                
    viterbi_acc = np.mean([1 if viterbi_preds[i] == ref_arcs[i] else 0 for i in range(n_obs)])
    dec_time = time.time() - t_dec_start
    
    # Calculate Route Continuity:
    # Proportion of transitions that are either same arc or directly connected in graph
    valid_trans_clf = 0
    valid_trans_vit = 0
    for t in range(n_obs - 1):
        a1_c, a2_c = clf_only_preds[t], clf_only_preds[t+1]
        if a1_c != -1 and a2_c != -1:
            if a1_c == a2_c or a2_c in out_edges.get(v_idx[a1_c], set()):
                valid_trans_clf += 1
                
        a1_v, a2_v = viterbi_preds[t], viterbi_preds[t+1]
        if a1_v != -1 and a2_v != -1:
            if a1_v == a2_v or a2_v in out_edges.get(v_idx[a1_v], set()):
                valid_trans_vit += 1
                
    cont_clf = valid_trans_clf / max(1, n_obs - 1)
    cont_vit = valid_trans_vit / max(1, n_obs - 1)
    
    results.append({
        'Trip ID': trip_id,
        'Observations': n_obs,
        'Classifier Only Acc': round(clf_acc * 100, 2),
        'HMM Viterbi Acc': round(viterbi_acc * 100, 2),
        'Accuracy Gain': round((viterbi_acc - clf_acc) * 100, 2),
        'Classifier Continuity': round(cont_clf * 100, 2),
        'Viterbi Continuity': round(cont_vit * 100, 2),
        'Decode Time (s)': round(dec_time, 3)
    })
    
    print(f"Trip {trip_id:<8} | Fixes: {n_obs:<5} | Clf: {clf_acc*100:6.2f}% | Viterbi: {viterbi_acc*100:6.2f}% | Gain: +{(viterbi_acc-clf_acc)*100:5.2f}% | Continuity: {cont_vit*100:5.1f}%")

print("="*80)
res_df = pd.DataFrame(results)

mean_clf = res_df['Classifier Only Acc'].mean()
mean_vit = res_df['HMM Viterbi Acc'].mean()
mean_gain = res_df['Accuracy Gain'].mean()
mean_cont_vit = res_df['Viterbi Continuity'].mean()

print(f"\nOVERALL 10-TRIP AVERAGE RESULTS:")
print(f"  Classifier-Only Accuracy:     {mean_clf:.2f}%")
print(f"  HMM Viterbi Route Accuracy:   {mean_vit:.2f}%")
print(f"  AVERAGE ACCURACY JUMP:        +{mean_gain:.2f}%")
print(f"  Viterbi Route Continuity:     {mean_cont_vit:.2f}%")

# Save outputs
res_df.to_csv('results/hmm_viterbi_evaluation_results.csv', index=False)
res_df.to_csv('PathSnap_ML_Hackathon_Team-11/results/hmm_viterbi_evaluation_results.csv', index=False)

# Plot comparison
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(res_df))
width = 0.35

rects1 = ax.bar(x - width/2, res_df['Classifier Only Acc'], width, label='Classifier-Only (Random Forest)', color='#EA4335')
rects2 = ax.bar(x + width/2, res_df['HMM Viterbi Acc'], width, label='Classifier + HMM Viterbi Decoder', color='#34A853')

ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('PathSnap Accuracy: Classifier-Only vs. Classifier + HMM Viterbi Across All 10 Test Trips', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(res_df['Trip ID'], rotation=30)
ax.set_ylim(40, 105)
ax.legend(fontsize=11)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('graphs/hmm_viterbi_accuracy_jump.png', dpi=150)
plt.savefig('PathSnap_ML_Hackathon_Team-11/graphs/hmm_viterbi_accuracy_jump.png', dpi=150)
plt.close()

print("\nSaved artifacts: results/hmm_viterbi_evaluation_results.csv, graphs/hmm_viterbi_accuracy_jump.png")
