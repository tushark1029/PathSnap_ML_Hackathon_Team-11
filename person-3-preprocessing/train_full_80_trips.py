import time
import os
import glob
import joblib
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, log_loss
)

def build_partition_candidates(partition='train', radius_m=50.0):
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
        
        # Ground truth mapping
        ref_arcs = []
        if route_df is not None and len(route_df) > 0:
            r_arcs = route_df['arc_id'].values
            ru, rv = arcs_df.iloc[r_arcs]['u'].values, arcs_df.iloc[r_arcs]['v'].values
            rx1, ry1 = node_x[ru], node_y[ru]
            rx2, ry2 = node_x[rv], node_y[rv]
            cur_r = 0
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
        else:
            ref_arcs = [-1] * n_obs
            
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
            cur_cand_list = []
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
                    cur_cand_list.append({
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
                        'flag_gap_gt_5s': 1 if dt[i] > 5.0 else 0
                    })
            prev_cand_arcs = set(c['candidate_arc_id'] for c in cur_cand_list)
            prev_cand_nodes_v = set(v_idx[c['candidate_arc_id']] for c in cur_cand_list)
            all_rows.extend(cur_cand_list)
            
    print(f"Generated {len(all_rows):,} candidate rows for {len(trip_dirs)} trips in {time.time()-t0:.2f}s")
    return pd.DataFrame(all_rows)

print("="*80)
print("STAGE 1: GENERATING CANDIDATES FOR ALL 80 TRAINING TRIPS AND ALL 10 TEST TRIPS")
print("="*80)

# 1. Generate full train (80 trips)
train_df = build_partition_candidates('train', radius_m=50.0)

# 2. Generate full test (10 trips)
test_df = build_partition_candidates('student_test', radius_m=50.0)

feature_cols = [
    'dist_to_candidate_arc_m',
    'proj_ratio',
    'arc_length_m',
    'arc_heading_deg',
    'heading_delta_deg',
    'connected_from_prev',
    'gps_speed_kmh',
    'gps_dist_m',
    'gps_time_diff_s',
    'flag_speed_anomaly',
    'flag_gap_gt_5s'
]

X_train = train_df[feature_cols].values
y_train = train_df['is_reference'].values

X_test = test_df[feature_cols].values
y_test = test_df['is_reference'].values

print(f"\nTraining Set Shape: {X_train.shape} | Positive reference labels: {y_train.sum():,} ({np.mean(y_train)*100:.2f}%)")
print(f"Test Set Shape:     {X_test.shape} | Positive reference labels: {y_test.sum():,} ({np.mean(y_test)*100:.2f}%)")

print("="*80)
print("STAGE 2: TRAINING AND EVALUATING ALL 5 MODELS ON COMPLETE 80 TRIPS")
print("="*80)

models = {
    "Logistic Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=12, 
        min_samples_leaf=25, 
        class_weight='balanced', 
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, 
        max_depth=14, 
        min_samples_leaf=15, 
        class_weight='balanced', 
        max_samples=0.25, # Bagging sub-sampling for fast, robust generalization
        n_jobs=-1, 
        random_state=42
    ),
    "K-Nearest Neighbors": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=15, weights='distance', n_jobs=-1))
    ]),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=100, 
        learning_rate=0.08, 
        class_weight='balanced', 
        random_state=42
    )
}

def compute_top1_accuracy(df_candidates, y_prob):
    df_eval = df_candidates[['trip_id', 'obs_idx', 'is_reference']].copy()
    df_eval['pred_score'] = y_prob
    idx_top1 = df_eval.groupby(['trip_id', 'obs_idx'])['pred_score'].idxmax()
    top1_df = df_eval.loc[idx_top1]
    return float(top1_df['is_reference'].mean())

results = []
trained_models = {}

print(f"{'Model':<25} | {'Train Time':<10} | {'Infer Time':<10} | {'F1-Score':<10} | {'ROC-AUC':<10} | {'Top-1 Acc':<10}")
print("-"*85)

for name, model in models.items():
    # If KNN, use stratified 50k reference sample to avoid hours of O(N*M) distance calculations
    if name == "K-Nearest Neighbors":
        sample_idx = np.random.RandomState(42).choice(len(X_train), size=min(50000, len(X_train)), replace=False)
        X_tr_fit = X_train[sample_idx]
        y_tr_fit = y_train[sample_idx]
    else:
        X_tr_fit = X_train
        y_tr_fit = y_train
        
    t_start = time.time()
    model.fit(X_tr_fit, y_tr_fit)
    train_time = time.time() - t_start
    
    t_start = time.time()
    # For KNN inference on 200k rows, sample 20k test fixes for timing or run chunked
    if name == "K-Nearest Neighbors":
        test_sub_idx = np.random.RandomState(42).choice(len(X_test), size=20000, replace=False)
        y_pred = model.predict(X_test[test_sub_idx])
        y_prob = model.predict_proba(X_test[test_sub_idx])[:, 1]
        infer_time = (time.time() - t_start) * (len(X_test) / 20000)
        # Scale top1 on subset
        top1_acc = compute_top1_accuracy(test_df.iloc[test_sub_idx], y_prob)
        y_test_eval = y_test[test_sub_idx]
    else:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        infer_time = time.time() - t_start
        top1_acc = compute_top1_accuracy(test_df, y_prob)
        y_test_eval = y_test
        
    acc = accuracy_score(y_test_eval, y_pred)
    prec = precision_score(y_test_eval, y_pred, zero_division=0)
    rec = recall_score(y_test_eval, y_pred, zero_division=0)
    f1 = f1_score(y_test_eval, y_pred, zero_division=0)
    auc = roc_auc_score(y_test_eval, y_prob)
    ll = log_loss(y_test_eval, y_prob)
    
    trained_models[name] = model
    
    results.append({
        'Model': name,
        'Train Time (s)': round(train_time, 2),
        'Inference Time (s)': round(infer_time, 2),
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
        'ROC-AUC': round(auc, 4),
        'Log Loss': round(ll, 4),
        'Top-1 Accuracy': round(top1_acc, 4)
    })
    
    print(f"{name:<25} | {train_time:<10.2f} | {infer_time:<10.2f} | {f1:<10.4f} | {auc:<10.4f} | {top1_acc:<10.4f}")

print("="*85)
results_df = pd.DataFrame(results)
print("\nFull Comparison Table Across All 80 Train Trips & All 10 Test Trips:")
print(results_df.to_string(index=False))

best_model_name = results_df.sort_values(by=['Top-1 Accuracy', 'F1-Score'], ascending=False).iloc[0]['Model']
print(f"\n=======================================================")
print(f">>> FINAL BEST MODEL ON FULL DATASET: {best_model_name} <<<")
print(f"=======================================================")

# Save outputs
results_df.to_csv('results/full_80_trips_model_comparison.csv', index=False)
results_df.to_csv('PathSnap_ML_Hackathon_Team-11/results/full_80_trips_model_comparison.csv', index=False)
joblib.dump(trained_models[best_model_name], 'models/best_model_full.joblib')
joblib.dump(trained_models[best_model_name], 'PathSnap_ML_Hackathon_Team-11/models/best_model_full.joblib')

# Save Visualization Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

bars1 = ax1.bar(results_df['Model'], results_df['Top-1 Accuracy'] * 100, color=['#4285F4', '#34A853', '#FBBC05', '#EA4335', '#9C27B0'])
ax1.set_title('Top-1 Observation Accuracy on ALL 10 Test Trips (%)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy (%)')
ax1.set_ylim(60, 100)
ax1.tick_params(axis='x', rotation=25)
ax1.grid(True, linestyle='--', alpha=0.5)
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')

bars2 = ax2.bar(results_df['Model'], results_df['F1-Score'], color=['#4285F4', '#34A853', '#FBBC05', '#EA4335', '#9C27B0'])
ax2.set_title('Candidate F1-Score Across ALL 10 Test Trips', fontsize=12, fontweight='bold')
ax2.set_ylabel('F1 Score')
ax2.set_ylim(0.5, 1.0)
ax2.tick_params(axis='x', rotation=25)
ax2.grid(True, linestyle='--', alpha=0.5)
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('graphs/full_80_trips_model_comparison_plot.png', dpi=150)
plt.savefig('PathSnap_ML_Hackathon_Team-11/graphs/full_80_trips_model_comparison_plot.png', dpi=150)
plt.close()

print("All outputs saved successfully.")
