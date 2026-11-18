import os
import pandas as pd
import numpy as np

csv_path = 'results/advanced_features_trip_breakdown.csv'
if not os.path.exists(csv_path):
    csv_path = 'PathSnap_ML_Hackathon_Team-11/results/advanced_features_trip_breakdown.csv'

df = pd.read_csv(csv_path)

print("="*85)
print("PATHSNAP TEAM 11: BEST MODEL ACCURACY & TRIP-BY-TRIP BENCHMARK")
print("="*85)
print(df[['trip_id', 'n_obs', 'rf_clf_acc', 'rf_hmm_acc', 'rf_continuity', 'blended_hmm_acc']].to_string(index=False))
print("-" * 85)
print(f"Overall RF Classifier Top-1 Accuracy:  {np.average(df['rf_clf_acc'], weights=df['n_obs']):.2f}%")
print(f"Overall RF + HMM Trajectory Accuracy:  {np.average(df['rf_hmm_acc'], weights=df['n_obs']):.2f}%  (Continuity: {np.average(df['rf_continuity'], weights=df['n_obs']):.2f}%)")
print(f"Overall Blended Ensemble + HMM:        {np.average(df['blended_hmm_acc'], weights=df['n_obs']):.2f}%  (Continuity: {np.average(df['blended_continuity'], weights=df['n_obs']):.2f}%)")
print("="*85)
