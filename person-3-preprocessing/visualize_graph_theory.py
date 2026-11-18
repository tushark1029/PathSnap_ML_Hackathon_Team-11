"""
visualize_graph_theory.py
=========================
PathSnap Team 11 - Graph Theory Visualization

Visualizes:
1. The Road Network Directed Graph G = (V, E) built with networkx.
2. Intersections (Nodes) and Directed Arcs (Edges).
3. The Viterbi Decoded Trajectory traversing the Graph.
"""

import os
import joblib
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from road_network_hmm_decoder import build_road_graph, NetworkHMMViterbiDecoder, ADVANCED_FEATURE_COLS

tid = '00000052'
td = f'docs/Student folder/student_test/{tid}'

nodes_df = pd.read_csv(f'{td}/{tid}.nodes', sep=r'\s+', header=None, names=['lon', 'lat'])
arcs_df = pd.read_csv(f'{td}/{tid}.arcs', sep=r'\s+', header=None, names=['u', 'v'])
track_df = pd.read_csv(f'{td}/{tid}.track', sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
route_df = pd.read_csv(f'{td}/{tid}.route', header=None, names=['arc_id'])

lat0 = np.mean(nodes_df['lat'])
G = build_road_graph(nodes_df, arcs_df, lat0)

print(f"Graph G for Trip {tid}:")
print(f"  Nodes (Intersections): {G.number_of_nodes()}")
print(f"  Edges (Directed Arcs): {G.number_of_edges()}")
print(f"  Is Strongly Connected: {nx.is_strongly_connected(G)}")
print(f"  Number of Strongly Connected Components: {nx.number_strongly_connected_components(G)}")

# Load model & test candidates
with open('data/candidates_test_10_trips_v2.pkl', 'rb') as f:
    test_df = pickle.load(f)

model = joblib.load('models/best_model_advanced_rf.joblib')
trip_cands = test_df[test_df['trip_id'] == tid].copy()
trip_cands['prob'] = model.predict_proba(trip_cands[ADVANCED_FEATURE_COLS].values)[:, 1]

decoder = NetworkHMMViterbiDecoder(G, arcs_df)
matched_route = decoder.decode(trip_cands)

# Node positions in metric space
pos = {n: (G.nodes[n]['x'], G.nodes[n]['y']) for n in G.nodes()}

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Subplot 1: Pure Graph Theory (Network Topology)
ax1 = axes[0]
ax1.set_title(f"Road Network Graph $G = (V, E)$\nNodes: {G.number_of_nodes()} Intersections | Edges: {G.number_of_edges()} Directed Arcs", fontsize=12, fontweight='bold')

# Draw all road edges
nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='#bdc3c7', arrows=True, arrowsize=8, width=0.8, alpha=0.7)
nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=12, node_color='#34495e', alpha=0.8)
ax1.set_xlabel("Local X Coordinate (meters)", fontsize=10)
ax1.set_ylabel("Local Y Coordinate (meters)", fontsize=10)
ax1.grid(True, linestyle=':', alpha=0.4)
ax1.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

# Subplot 2: Trajectory Map-Matching on Graph
ax2 = axes[1]
ax2.set_title(f"Trajectory Map-Matching on Graph\nRaw GPS Points vs. Viterbi Path on Graph", fontsize=12, fontweight='bold')

# Background graph
nx.draw_networkx_edges(G, pos, ax=ax2, edge_color='#ecf0f1', arrows=False, width=1.0)

# Raw GPS points
m_per_deg_lat = 111139.0
m_per_deg_lon = 111139.0 * np.cos(np.radians(lat0))
tx = track_df['lon'].values * m_per_deg_lon
ty = track_df['lat'].values * m_per_deg_lat
ax2.scatter(tx, ty, c='#e74c3c', s=15, label='Raw GPS Trajectory Fixes', zorder=4, alpha=0.7)

# Matched edges on the graph
matched_edges = []
for aid in matched_route:
    u = arcs_df.iloc[aid]['u']
    v = arcs_df.iloc[aid]['v']
    matched_edges.append((u, v))

nx.draw_networkx_edges(G, pos, edgelist=matched_edges, ax=ax2, edge_color='#27ae60', width=3.5, arrows=True, arrowsize=14, label='HMM Viterbi Matched Path on Graph')

ax2.set_xlabel("Local X Coordinate (meters)", fontsize=10)
ax2.set_ylabel("Local Y Coordinate (meters)", fontsize=10)
ax2.legend(loc='upper right', frameon=True, fontsize=10)
ax2.grid(True, linestyle=':', alpha=0.4)
ax2.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

plt.tight_layout()
os.makedirs('graphs', exist_ok=True)
os.makedirs('PathSnap_ML_Hackathon_Team-11/graphs', exist_ok=True)
plt.savefig('graphs/road_network_graph_theory_visualization.png', dpi=150)
plt.savefig('PathSnap_ML_Hackathon_Team-11/graphs/road_network_graph_theory_visualization.png', dpi=150)
plt.close()

print("Graph Theory visualization created at graphs/road_network_graph_theory_visualization.png")
