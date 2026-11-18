import time
import os
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
    roc_auc_score, log_loss, confusion_matrix
)

# 1. Load Data
# Use generated candidate datasets: Trips 1, 3, 4, 5, 6 as Train, Trip 0 (Student Test) as Test
from subprocess import check_output

print("Generating/loading training candidate dataset...")
# Let's import the builder from our previous code
# from generate_candidate_features import build_candidate_dataset

train_trips = ['00000001', '00000003', '00000004', '00000005', '00000006']
test_trips = ['00000000'] # Student test

# We also need candidate builder for student_test partition
def build_partition_candidates(trip_ids, partition='train', radius_m=50.0):
    all_rows = []
    base_dir = f'docs/Student folder/{partition}'
    
    for trip_id in trip_ids:
        td = os.path.join(base_dir, trip_id)
        if not os.path.isdir(td): continue
        track_df = pd.read_csv(f'{td}/{trip_id}.track', sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
        nodes_df = pd.read_csv(f'{td}/{trip_id}.nodes', sep=r'\s+', header=None, names=['lon', 'lat'])
        arcs_df = pd.read_csv(f'{td}/{trip_id}.arcs', sep=r'\s+', header=None, names=['u', 'v'])
        route_df = pd.read_csv(f'{td}/{trip_id}.route', header=None, names=['arc_id'])
        
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
        
        r_arcs = route_df['arc_id'].values
        ru, rv = arcs_df.iloc[r_arcs]['u'].values, arcs_df.iloc[r_arcs]['v'].values
        rx1, ry1 = node_x[ru], node_y[ru]
        rx2, ry2 = node_x[rv], node_y[rv]
        
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
                        'gps_heading_deg': gps_headings[i],
                        'flag_speed_anomaly': 1 if speed_ms[i] > 50.0 else 0,
                        'flag_gap_gt_5s': 1 if dt[i] > 5.0 else 0
                    })
            prev_cand_arcs = set(c['candidate_arc_id'] for c in cur_cand_list)
            prev_cand_nodes_v = set(v_idx[c['candidate_arc_id']] for c in cur_cand_list)
            all_rows.extend(cur_cand_list)
    return pd.DataFrame(all_rows)

print("Building training set (5 trips)...")
t0 = time.time()
train_df = build_partition_candidates(train_trips, 'train', radius_m=50.0)
print(f"Training set: {len(train_df):,} rows ({time.time()-t0:.2f}s)")

print("Building test set (Student test Trip 00000000)...")
t0 = time.time()
test_df = build_partition_candidates(test_trips, 'student_test', radius_m=50.0)
print(f"Test set: {len(test_df):,} rows ({time.time()-t0:.2f}s)")

# Define Features and Target
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

print(f"Features: {feature_cols}")
print(f"Train class balance: {np.bincount(y_train)} (positive: {np.mean(y_train)*100:.2f}%)")
print(f"Test class balance:  {np.bincount(y_test)} (positive: {np.mean(y_test)*100:.2f}%)")

# Define the 5 Models
models = {
    "Logistic Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10, 
        min_samples_leaf=20, 
        class_weight='balanced', 
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, 
        max_depth=12, 
        min_samples_leaf=10, 
        class_weight='balanced', 
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

# Function to calculate Observation-level Top-1 Accuracy
def compute_top1_accuracy(df_candidates, y_prob):
    df_eval = df_candidates[['trip_id', 'obs_idx', 'is_reference']].copy()
    df_eval['pred_score'] = y_prob
    
    # For each (trip_id, obs_idx), find candidate with max pred_score
    idx_top1 = df_eval.groupby(['trip_id', 'obs_idx'])['pred_score'].idxmax()
    top1_df = df_eval.loc[idx_top1]
    
    # Top-1 accuracy is proportion of times the top candidate has is_reference == 1
    top1_acc = top1_df['is_reference'].mean()
    return top1_acc

results = []
trained_models = {}

print("\n" + "="*80)
print(f"{'Model':<25} | {'Train Time':<10} | {'Infer Time':<10} | {'F1-Score':<10} | {'ROC-AUC':<10} | {'Top-1 Acc':<10}")
print("="*80)

for name, model in models.items():
    # 1. Training
    t_start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t_start
    
    # 2. Inference
    t_start = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    infer_time = time.time() - t_start
    
    # 3. Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    ll = log_loss(y_test, y_prob)
    
    top1_acc = compute_top1_accuracy(test_df, y_prob)
    
    trained_models[name] = model
    
    results.append({
        'Model': name,
        'Train Time (s)': round(train_time, 3),
        'Inference Time (s)': round(infer_time, 3),
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
        'ROC-AUC': round(auc, 4),
        'Log Loss': round(ll, 4),
        'Top-1 Accuracy': round(top1_acc, 4)
    })
    
    print(f"{name:<25} | {train_time:<10.3f} | {infer_time:<10.3f} | {f1:<10.4f} | {auc:<10.4f} | {top1_acc:<10.4f}")

print("="*80)
results_df = pd.DataFrame(results)
print("\nFull Comparison Table:")
print(results_df.to_string(index=False))

# Identify Best Model
best_model_name = results_df.sort_values(by=['Top-1 Accuracy', 'F1-Score'], ascending=False).iloc[0]['Model']
print(f"\n>>> BEST MODEL SELECTED: {best_model_name} <<<")

# Save Comparison Results
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('PathSnap_ML_Hackathon_Team-11/results', exist_ok=True)
os.makedirs('PathSnap_ML_Hackathon_Team-11/models', exist_ok=True)

results_df.to_csv('results/model_comparison_results.csv', index=False)
results_df.to_csv('PathSnap_ML_Hackathon_Team-11/results/model_comparison_results.csv', index=False)

# Save the best model
joblib.dump(trained_models[best_model_name], f'models/best_model_{best_model_name.replace(" ", "_").lower()}.joblib')
joblib.dump(trained_models[best_model_name], f'PathSnap_ML_Hackathon_Team-11/models/best_model_{best_model_name.replace(" ", "_").lower()}.joblib')

# Save Visualization Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Top-1 Accuracy Bar Plot
bars1 = ax1.bar(results_df['Model'], results_df['Top-1 Accuracy'] * 100, color=['#4285F4', '#34A853', '#FBBC05', '#EA4335', '#9C27B0'])
ax1.set_title('Observation-Level Top-1 Accuracy (%)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Accuracy (%)')
ax1.set_ylim(50, 100)
ax1.tick_params(axis='x', rotation=25)
ax1.grid(True, linestyle='--', alpha=0.5)
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')

# F1-Score vs Inference Time Scatter
ax2.scatter(results_df['Inference Time (s)'], results_df['F1-Score'], s=200, c=['#4285F4', '#34A853', '#FBBC05', '#EA4335', '#9C27B0'], edgecolors='black')
for i, txt in enumerate(results_df['Model']):
    ax2.annotate(txt, (results_df['Inference Time (s)'][i], results_df['F1-Score'][i]), xytext=(5, 5), textcoords='offset points', fontweight='bold')
ax2.set_title('F1-Score vs. Inference Latency', fontsize=13, fontweight='bold')
ax2.set_xlabel('Inference Time (seconds)')
ax2.set_ylabel('Candidate F1-Score')
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('graphs/model_comparison_plot.png', dpi=150)
plt.savefig('PathSnap_ML_Hackathon_Team-11/graphs/model_comparison_plot.png', dpi=150)
plt.close()

print("Artifacts saved: model_comparison_results.csv, model_comparison_plot.png, best_model joblib.")
