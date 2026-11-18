import time
import os
import pickle
import joblib
import optuna
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss

# Suppress Optuna verbose logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("="*85)
print("OPTUNA HYPERPARAMETER TUNING: ENRICHED 17-FEATURE RANDOM FOREST MODEL")
print("="*85)

os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('graphs', exist_ok=True)
os.makedirs('PathSnap_ML_Hackathon_Team-11/models', exist_ok=True)
os.makedirs('PathSnap_ML_Hackathon_Team-11/results', exist_ok=True)
os.makedirs('PathSnap_ML_Hackathon_Team-11/graphs', exist_ok=True)

ADVANCED_FEATURE_COLS = [
    'dist_to_candidate_arc_m', 'proj_ratio', 'arc_length_m', 'arc_heading_deg',
    'heading_delta_deg', 'connected_from_prev', 'gps_speed_kmh', 'gps_dist_m',
    'gps_time_diff_s', 'flag_speed_anomaly', 'flag_gap_gt_5s',
    'cos_heading_diff', 'dist_rank', 'dist_ratio_to_min', 'dist_diff_from_min',
    'dist_to_u_m', 'dist_to_v_m'
]

# -----------------------------------------------------------------------------
# 1. LOAD ENRICHED V2 CANDIDATE DATASETS
# -----------------------------------------------------------------------------
TRAIN_CACHE = 'data/candidates_train_80_trips_v2.pkl'
TEST_CACHE = 'data/candidates_test_10_trips_v2.pkl'

print(f"Loading enriched datasets from {TRAIN_CACHE} and {TEST_CACHE}...")
t0 = time.time()
with open(TRAIN_CACHE, 'rb') as f:
    train_df = pickle.load(f)
with open(TEST_CACHE, 'rb') as f:
    test_df = pickle.load(f)

print(f"Loaded in {time.time()-t0:.2f}s:")
print(f"  Training Set: {len(train_df):,} rows ({train_df['trip_id'].nunique()} trips, {len(ADVANCED_FEATURE_COLS)} features)")
print(f"  Test Set:     {len(test_df):,} rows ({test_df['trip_id'].nunique()} trips)")

# -----------------------------------------------------------------------------
# 2. ZERO-DATA-LEAKAGE GROUP VALIDATION SPLIT
# -----------------------------------------------------------------------------
unique_train_trips = np.array(sorted(train_df['trip_id'].unique()))
np.random.seed(42)
shuffled_trips = np.random.permutation(unique_train_trips)

n_val = int(len(shuffled_trips) * 0.20)
val_trips = set(shuffled_trips[:n_val])
tr_trips = set(shuffled_trips[n_val:])

print(f"Group Split: {len(tr_trips)} Training Trips | {len(val_trips)} Unseen Validation Trips")

tr_mask = train_df['trip_id'].isin(tr_trips)
val_mask = train_df['trip_id'].isin(val_trips)

X_tr = train_df.loc[tr_mask, ADVANCED_FEATURE_COLS].values
y_tr = train_df.loc[tr_mask, 'is_reference'].values

X_val = train_df.loc[val_mask, ADVANCED_FEATURE_COLS].values
y_val = train_df.loc[val_mask, 'is_reference'].values
val_sub_df = train_df.loc[val_mask, ['trip_id', 'obs_idx', 'candidate_arc_id', 'is_reference']].copy()

def fast_compute_top1_accuracy(proba):
    val_sub_df['score'] = proba
    idx_max = val_sub_df.groupby(['trip_id', 'obs_idx'])['score'].idxmax()
    top1_correct = val_sub_df.loc[idx_max, 'is_reference'].sum()
    return top1_correct / len(idx_max)

# -----------------------------------------------------------------------------
# 3. DEFINE OPTUNA OBJECTIVE FUNCTION
# -----------------------------------------------------------------------------
trial_history = []

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 80, 180, step=20)
    max_depth = trial.suggest_int('max_depth', 10, 22, step=2)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 4, 16, step=2)
    min_samples_split = trial.suggest_int('min_samples_split', 6, 24, step=3)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 0.4, 0.6])
    max_samples = trial.suggest_float('max_samples', 0.30, 0.60, step=0.10)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    class_weight = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample'])
    
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_features=max_features,
        max_samples=max_samples,
        criterion=criterion,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=42
    )
    
    t_start = time.time()
    rf.fit(X_tr, y_tr)
    fit_duration = time.time() - t_start
    
    val_probs = rf.predict_proba(X_val)[:, 1]
    top1_acc = fast_compute_top1_accuracy(val_probs)
    val_roc = roc_auc_score(y_val, val_probs)
    val_preds = (val_probs >= 0.5).astype(int)
    val_f1 = f1_score(y_val, val_preds)
    val_ll = log_loss(y_val, val_probs)
    
    trial_history.append({
        'trial_number': trial.number,
        'top1_accuracy': round(top1_acc, 4),
        'roc_auc': round(val_roc, 4),
        'f1_score': round(val_f1, 4),
        'log_loss': round(val_ll, 4),
        'fit_time_s': round(fit_duration, 2),
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_leaf': min_samples_leaf,
        'min_samples_split': min_samples_split,
        'max_features': str(max_features),
        'max_samples': max_samples,
        'criterion': criterion,
        'class_weight': class_weight
    })
    
    print(f"Trial #{trial.number:02d} | Top-1: {top1_acc*100:.2f}% | ROC-AUC: {val_roc:.4f} | F1: {val_f1:.4f} | Depth: {max_depth} | Feat: {max_features} | Leaf: {min_samples_leaf} | {fit_duration:.1f}s")
    
    return top1_acc

# -----------------------------------------------------------------------------
# 4. EXECUTE OPTUNA OPTIMIZATION STUDY
# -----------------------------------------------------------------------------
N_TRIALS = 15
print(f"\nStarting Optuna Study ({N_TRIALS} Bayesian TPE Trials on 17 Enriched Features)...")
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="maximize", sampler=sampler, study_name="rf_v2_hyperparameter_tuning")

study_start = time.time()
study.optimize(objective, n_trials=N_TRIALS)
total_study_time = time.time() - study_start

print(f"\nOptuna Study completed {N_TRIALS} trials in {total_study_time:.2f}s ({total_study_time/60:.2f} min)")
print("="*85)
print(f">>> BEST TRIAL #{study.best_trial.number} <<<")
print(f"Best Validation Top-1 Accuracy: {study.best_value * 100:.2f}%")
print("Best Hyperparameters:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")
print("="*85)

# Save trials DataFrame
trials_df = pd.DataFrame(trial_history)
trials_df = trials_df.sort_values(by='top1_accuracy', ascending=False).reset_index(drop=True)
trials_df.to_csv('results/optuna_v2_tuning_trials.csv', index=False)
trials_df.to_csv('PathSnap_ML_Hackathon_Team-11/results/optuna_v2_tuning_trials.csv', index=False)

# -----------------------------------------------------------------------------
# 5. RETRAIN BEST TUNED MODEL ON ALL 80 TRAINING TRIPS
# -----------------------------------------------------------------------------
print("\n" + "="*85)
print("STAGE 5: FULL RETRAINING OF OPTUNA-OPTIMIZED V2 MODEL ON ALL 80 TRAINING TRIPS")
print("="*85)

best_params = study.best_params.copy()
best_params['n_jobs'] = -1
best_params['random_state'] = 42

X_train_full = train_df[ADVANCED_FEATURE_COLS].values
y_train_full = train_df['is_reference'].values

t0_full = time.time()
best_optuna_rf_v2 = RandomForestClassifier(**best_params)
best_optuna_rf_v2.fit(X_train_full, y_train_full)
full_train_duration = time.time() - t0_full
print(f"Full Retraining Completed in {full_train_duration:.2f}s")

# Serialize model
joblib.dump(best_optuna_rf_v2, 'models/best_tuned_rf_v2.joblib')
joblib.dump(best_optuna_rf_v2, 'PathSnap_ML_Hackathon_Team-11/models/best_tuned_rf_v2.joblib')
print("Model saved to models/best_tuned_rf_v2.joblib")

# -----------------------------------------------------------------------------
# 6. BENCHMARK ON ALL 10 STUDENT-TEST TRIPS: UNTUNED V2 vs. OPTUNA TUNED V2
# -----------------------------------------------------------------------------
print("\n" + "="*85)
print("STAGE 6: BENCHMARKING ON ALL 10 STUDENT-TEST TRIPS")
print("="*85)

previous_rf = joblib.load('models/best_model_advanced_rf.joblib')
test_trips = sorted(test_df['trip_id'].unique())

def evaluate_trip_with_hmm(trip_id, clf_model):
    trip_cands = test_df[test_df['trip_id'] == trip_id].copy()
    if len(trip_cands) == 0:
        return None
        
    td = f'docs/Student folder/student_test/{trip_id}'
    nodes_df = pd.read_csv(f'{td}/{trip_id}.nodes', sep=r'\s+', header=None, names=['lon', 'lat'])
    arcs_df = pd.read_csv(f'{td}/{trip_id}.arcs', sep=r'\s+', header=None, names=['u', 'v'])
    route_df = pd.read_csv(f'{td}/{trip_id}.route', header=None, names=['arc_id'])
    track_df = pd.read_csv(f'{td}/{trip_id}.track', sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
    
    probs = clf_model.predict_proba(trip_cands[ADVANCED_FEATURE_COLS].values)[:, 1]
    trip_cands['prob'] = probs
    
    obs_groups = [group for _, group in trip_cands.groupby('obs_idx')]
    T = len(obs_groups)
    
    clf_top1 = []
    for g in obs_groups:
        best_cand = g.loc[g['prob'].idxmax()]
        clf_top1.append(int(best_cand['candidate_arc_id']))
        
    # Viterbi DP
    eps = 1e-6
    first_cands = obs_groups[0]['candidate_arc_id'].values
    first_probs = obs_groups[0]['prob'].values
    first_dists = obs_groups[0]['dist_to_candidate_arc_m'].values
    first_emiss = np.log(first_probs + eps) - 0.5 * (first_dists / 15.0)**2
    
    dp = {arc: (first_emiss[i], [arc]) for i, arc in enumerate(first_cands)}
    
    for t in range(1, T):
        curr_cands = obs_groups[t]['candidate_arc_id'].values
        curr_probs = obs_groups[t]['prob'].values
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
    clf_top1 = np.array(clf_top1)
    
    # Ground truth reference arcs
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
    
    clf_acc = np.mean(clf_top1 == true_obs_arcs)
    vit_acc = np.mean(viterbi_route == true_obs_arcs)
    
    cont = 0
    for i in range(len(viterbi_route)-1):
        a1, a2 = viterbi_route[i], viterbi_route[i+1]
        if a1 == a2 or arcs_df.iloc[a1]['v'] == arcs_df.iloc[a2]['u']:
            cont += 1
    continuity = cont / (len(viterbi_route)-1) if len(viterbi_route) > 1 else 1.0
    
    return {
        'trip_id': trip_id,
        'n_obs': T,
        'clf_acc': clf_acc,
        'vit_acc': vit_acc,
        'continuity': continuity
    }

comparison_rows = []
for tid in test_trips:
    base_res = evaluate_trip_with_hmm(tid, previous_rf)
    opt_res = evaluate_trip_with_hmm(tid, best_optuna_rf_v2)
    
    comparison_rows.append({
        'trip_id': tid,
        'n_obs': base_res['n_obs'],
        'prev_clf_acc': round(base_res['clf_acc'] * 100, 2),
        'optuna_clf_acc': round(opt_res['clf_acc'] * 100, 2),
        'clf_gain': round((opt_res['clf_acc'] - base_res['clf_acc']) * 100, 2),
        'prev_hmm_acc': round(base_res['vit_acc'] * 100, 2),
        'optuna_hmm_acc': round(opt_res['vit_acc'] * 100, 2),
        'hmm_gain': round((opt_res['vit_acc'] - base_res['vit_acc']) * 100, 2),
        'optuna_continuity': round(opt_res['continuity'] * 100, 2)
    })

comp_df = pd.DataFrame(comparison_rows)

total_obs = comp_df['n_obs'].sum()
avg_prev_clf = np.average(comp_df['prev_clf_acc'], weights=comp_df['n_obs'])
avg_opt_clf = np.average(comp_df['optuna_clf_acc'], weights=comp_df['n_obs'])
avg_prev_hmm = np.average(comp_df['prev_hmm_acc'], weights=comp_df['n_obs'])
avg_opt_hmm = np.average(comp_df['optuna_hmm_acc'], weights=comp_df['n_obs'])
avg_cont = np.average(comp_df['optuna_continuity'], weights=comp_df['n_obs'])

print("\n" + "="*85)
print("TRIP-BY-TRIP BENCHMARK: PREVIOUS V2 RANDOM FOREST vs. OPTUNA TUNED V2 RANDOM FOREST")
print("="*85)
print(comp_df.to_string(index=False))

print("-" * 85)
print(f"Overall Classifier Accuracy: Previous = {avg_prev_clf:.2f}%  -->  Optuna Tuned = {avg_opt_clf:.2f}% (Gain: {avg_opt_clf - avg_prev_clf:+.2f}%)")
print(f"Overall HMM Viterbi Accuracy: Previous = {avg_prev_hmm:.2f}%  -->  Optuna Tuned = {avg_opt_hmm:.2f}% (Gain: {avg_opt_hmm - avg_prev_hmm:+.2f}%)")
print(f"Overall Route Continuity:    {avg_cont:.2f}%")
print("="*85)

comp_df.to_csv('results/optuna_v2_vs_baseline_comparison.csv', index=False)
comp_df.to_csv('PathSnap_ML_Hackathon_Team-11/results/optuna_v2_vs_baseline_comparison.csv', index=False)

# -----------------------------------------------------------------------------
# 7. GENERATE VISUALIZATION PLOTS
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

trials_sorted = pd.DataFrame(trial_history).sort_values(by='trial_number')
ax1 = axes[0]
ax1.plot(trials_sorted['trial_number'], trials_sorted['top1_accuracy'] * 100, marker='o', color='#1f77b4', linewidth=2, label='Trial Top-1 Acc')
running_max = np.maximum.accumulate(trials_sorted['top1_accuracy'] * 100)
ax1.plot(trials_sorted['trial_number'], running_max, color='#d62728', linestyle='--', linewidth=2, label='Best So Far')
ax1.set_xlabel('Trial Number', fontsize=11, fontweight='bold')
ax1.set_ylabel('Validation Top-1 Accuracy (%)', fontsize=11, fontweight='bold')
ax1.set_title('Optuna Optimization History (17 Enriched Features)', fontsize=12, fontweight='bold')
ax1.legend(loc='lower right', frameon=True)
ax1.grid(True, linestyle='--', alpha=0.5)

ax2 = axes[1]
x = np.arange(len(comp_df))
width = 0.35
b1 = ax2.bar(x - width/2, comp_df['prev_hmm_acc'], width, label=f'Previous RF+HMM ({avg_prev_hmm:.2f}%)', color='#708090', alpha=0.85, edgecolor='black')
b2 = ax2.bar(x + width/2, comp_df['optuna_hmm_acc'], width, label=f'Optuna Tuned RF+HMM ({avg_opt_hmm:.2f}%)', color='#2ca02c', alpha=0.85, edgecolor='black')

ax2.set_xticks(x)
ax2.set_xticklabels([f"Trip {tid}" for tid in comp_df['trip_id']], rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('Trajectory Accuracy (%)', fontsize=11, fontweight='bold')
ax2.set_title(f'Test Trajectory Accuracy (Optuna Tuned: {avg_opt_hmm:.2f}%)', fontsize=12, fontweight='bold')
ax2.legend(loc='lower left', frameon=True)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('graphs/optuna_v2_tuning_comparison.png', dpi=150)
plt.savefig('PathSnap_ML_Hackathon_Team-11/graphs/optuna_v2_tuning_comparison.png', dpi=150)
plt.close()

print("Saved plots to graphs/optuna_v2_tuning_comparison.png")
print("Optuna hyperparameter tuning on enriched Random Forest successfully finished!")
