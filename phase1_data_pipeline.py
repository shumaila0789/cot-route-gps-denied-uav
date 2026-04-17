"""
CoT-Route — Phase 1 (Updated): Real TartanAir Data Pipeline
=============================================================
Updated to correctly read the downloaded TartanAir folder structure.

HOW TO RUN:
  Replace your existing phase1_data_pipeline.py with this file.
  Then run: python phase1_data_pipeline.py

All outputs saved in the same folder as this script.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import networkx as nx
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Project paths ─────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.resolve()
DATA_DIR      = BASE_DIR / "data"
TARTANAIR_DIR = DATA_DIR / "tartanair"
PROCESSED_DIR = DATA_DIR / "processed"
LABELS_DIR    = DATA_DIR / "labels"
GRAPHS_DIR    = DATA_DIR / "graphs"
FIGURES_DIR   = BASE_DIR / "figures"
LOGS_DIR      = BASE_DIR / "logs"

for d in [PROCESSED_DIR, LABELS_DIR, GRAPHS_DIR, FIGURES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
KEYFRAME_STRIDE   = 10      # every Nth frame as graph node
MAX_EDGE_DIST_M   = 1.5     # max distance (m) to connect two nodes
OBSTACLE_THRESH_M = 2.0     # depth below this = risky
MAX_FRAMES_PER_SEQ= 500     # cap per sequence to keep runtime reasonable

# Environments — split into train and val
TRAIN_ENVS = [
    "abandonedfactory",
    "abandonedfactory_night",
    "hospital",
    "office",
    "office2",
    "amusement",
    "carwelding",
    "japanesealley",
    "seasonsforest",
    "oldtown",
]
VAL_ENVS = [
    "gascola",
    "ocean",
    "westerndesert",
]

# ── Logger ────────────────────────────────────────────────────────────────────
log_lines = []

def log(msg, level="INFO"):
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    print(line)
    log_lines.append(line)

def save_log():
    path = LOGS_DIR / "phase1_pipeline.log"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Discover sequences using actual downloaded structure
#  Real structure:
#    tartanair/{env}/Easy/image_left/{env}/Easy/P000/image_left/*.png
#    tartanair/{env}/Easy/image_left/{env}/Easy/P000/pose_left.txt
#    tartanair/{env}/Easy/depth_left/{env}/Easy/P000/*.npy
# ══════════════════════════════════════════════════════════════════════════════
def find_sequence_paths(env_name):
    sequences = []
    env_dir   = TARTANAIR_DIR / env_name / "Easy"
    if not env_dir.exists():
        return sequences

    img_base   = env_dir / "image_left" / env_name / "Easy"
    depth_base = env_dir / "depth_left" / env_name / "Easy"

    if not img_base.exists():
        return sequences

    traj_dirs = sorted([d for d in img_base.iterdir() if d.is_dir()])

    for traj_dir in traj_dirs:
        traj_name = traj_dir.name
        rgb_dir   = traj_dir / "image_left"
        pose_file = traj_dir / "pose_left.txt"
        # Depth files live inside an extra depth_left subfolder:
        # depth_base/P000/depth_left/000000_left_depth.npy
        if depth_base.exists():
            candidate = depth_base / traj_name / "depth_left"
            depth_dir = candidate if candidate.exists() else depth_base / traj_name
        else:
            depth_dir = None

        if not rgb_dir.exists() or not pose_file.exists():
            continue

        img_files = sorted(rgb_dir.glob("*.png"))
        if len(img_files) == 0:
            continue

        sequences.append({
            "env":       env_name,
            "traj":      traj_name,
            "rgb_dir":   rgb_dir,
            "depth_dir": depth_dir,
            "pose_file": pose_file,
            "n_images":  len(img_files),
        })

    return sequences

def discover_all_sequences():
    log("=" * 62)
    log("STEP 1 — Discovering TartanAir sequences")
    log("=" * 62)

    train_seqs, val_seqs = [], []

    for env in TRAIN_ENVS:
        seqs = find_sequence_paths(env)
        if seqs:
            log(f"  [train] {env}: {len(seqs)} trajectories, "
                f"{sum(s['n_images'] for s in seqs)} frames")
            train_seqs.extend(seqs)
        else:
            log(f"  [skip]  {env}: not found", "WARN")

    for env in VAL_ENVS:
        seqs = find_sequence_paths(env)
        if seqs:
            log(f"  [val]   {env}: {len(seqs)} trajectories")
            val_seqs.extend(seqs)

    log(f"Train sequences: {len(train_seqs)}")
    log(f"Val sequences  : {len(val_seqs)}")
    return train_seqs, val_seqs

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Feature computation
# ══════════════════════════════════════════════════════════════════════════════
def load_poses(pose_path):
    poses = []
    with open(pose_path, "r") as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) >= 7:
                poses.append([float(v) for v in vals[:7]])
    return np.array(poses) if poses else np.zeros((0, 7))

def compute_texture_richness(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.var(np.sqrt(gx**2 + gy**2)))

def compute_obstacle_risk(depth):
    if depth is None:
        return 0.5
    h   = depth.shape[0]
    roi = depth[:h * 3 // 4, :]
    valid = roi[(roi > 0.1) & (roi < 50.0)]
    if len(valid) == 0:
        return 0.5
    return float(np.clip(np.sum(valid < OBSTACLE_THRESH_M) / len(valid), 0, 1))

def compute_uncertainty(texture):
    norm = np.clip(np.log1p(texture) / np.log1p(50000.0), 0, 1)
    return float(np.clip(1.0 - norm, 0.0, 1.0))

def load_depth(depth_dir, frame_idx):
    if depth_dir is None:
        return None
    base = Path(depth_dir)
    # Try direct path first, then depth_left subfolder as fallback
    for candidate_dir in [base, base / "depth_left"]:
        path = candidate_dir / f"{frame_idx:06d}_left_depth.npy"
        if path.exists():
            try:
                return np.load(str(path)).astype(np.float32)
            except Exception:
                return None
    return None

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Process sequences
# ══════════════════════════════════════════════════════════════════════════════
def process_sequence(seq_info, global_offset=0):
    img_files = sorted(Path(seq_info["rgb_dir"]).glob("*.png"))
    poses     = load_poses(seq_info["pose_file"])
    n_total   = min(len(img_files), len(poses), MAX_FRAMES_PER_SEQ)

    if n_total == 0:
        return []

    records = []
    for frame_idx in range(0, n_total, KEYFRAME_STRIDE):
        img = cv2.imread(str(img_files[frame_idx]))
        if img is None:
            continue

        pose    = poses[frame_idx]
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        depth   = load_depth(seq_info["depth_dir"], frame_idx)
        texture = compute_texture_richness(gray)
        risk    = compute_obstacle_risk(depth)
        uncert  = compute_uncertainty(texture)

        if frame_idx > 0:
            prev  = poses[max(0, frame_idx - KEYFRAME_STRIDE)]
            speed = float(np.linalg.norm((pose[:3] - prev[:3]) / (KEYFRAME_STRIDE * 0.1)))
        else:
            speed = 0.0

        records.append({
            "keyframe_idx":     global_offset + len(records),
            "frame_idx":        frame_idx,
            "env":              seq_info["env"],
            "traj":             seq_info["traj"],
            "pose":             pose.tolist(),
            "position":         pose[:3].tolist(),
            "texture_richness": round(texture, 4),
            "semantic_risk":    round(risk,    4),
            "uncertainty":      round(uncert,  4),
            "imu_text":         f"forward_velocity: {speed:.2f} m/s, "
                                f"position: ({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}) m",
            "rgb_path":         str(img_files[frame_idx]),
            "depth_available":  depth is not None,
        })
    return records

def process_all_sequences(train_seqs, val_seqs):
    log("=" * 62)
    log("STEP 2 — Processing keyframes")
    log("=" * 62)

    all_records, train_records, val_records = [], [], []

    for seq in tqdm(train_seqs, desc="Train"):
        recs = process_sequence(seq, global_offset=len(all_records))
        train_records.extend(recs)
        all_records.extend(recs)
        if recs:
            tqdm.write(f"  {seq['env']}/{seq['traj']}: {len(recs)} keyframes")

    for seq in tqdm(val_seqs, desc="Val"):
        recs = process_sequence(seq, global_offset=len(all_records))
        val_records.extend(recs)
        all_records.extend(recs)

    log(f"Total keyframes: {len(all_records)} "
        f"(train: {len(train_records)}, val: {len(val_records)})")

    with open(PROCESSED_DIR / "keyframe_records.json", "w") as f:
        json.dump(all_records, f, indent=2)
    with open(PROCESSED_DIR / "train_records.json", "w") as f:
        json.dump(train_records, f, indent=2)
    with open(PROCESSED_DIR / "val_records.json", "w") as f:
        json.dump(val_records, f, indent=2)

    return all_records, train_records, val_records

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Build graphs
# ══════════════════════════════════════════════════════════════════════════════
def build_all_graphs(all_records):
    log("=" * 62)
    log("STEP 3 — Building navigation graphs")
    log("=" * 62)

    alpha, beta, gamma = 0.4, 0.35, 0.25
    envs   = list(set(r["env"] for r in all_records))
    graphs = {}

    for env in sorted(envs):
        env_recs = [r for r in all_records if r["env"] == env]
        if len(env_recs) < 2:
            continue

        G = nx.DiGraph()
        for r in env_recs:
            G.add_node(r["keyframe_idx"],
                       frame_idx=r["frame_idx"], position=r["position"],
                       risk=r["semantic_risk"], uncertainty=r["uncertainty"],
                       env=r["env"], traj=r["traj"])

        # Connect within same trajectory (sequential edges)
        traj_groups = {}
        for r in env_recs:
            traj_groups.setdefault(r["traj"], []).append(r)

        for traj_recs in traj_groups.values():
            for i in range(len(traj_recs)):
                for j in range(i + 1, min(i + 6, len(traj_recs))):
                    ri, rj = traj_recs[i], traj_recs[j]
                    dist = float(np.linalg.norm(
                        np.array(ri["position"]) - np.array(rj["position"])))
                    if dist <= MAX_EDGE_DIST_M:
                        s    = (ri["semantic_risk"] + rj["semantic_risk"]) / 2
                        u    = (ri["uncertainty"]   + rj["uncertainty"])   / 2
                        cost = alpha * dist + beta * s + gamma * u
                        for a, b in [(ri["keyframe_idx"], rj["keyframe_idx"]),
                                     (rj["keyframe_idx"], ri["keyframe_idx"])]:
                            G.add_edge(a, b, distance=round(dist,4),
                                       semantic_risk=round(s,4),
                                       uncertainty=round(u,4),
                                       cost=round(cost,4))

        # ── Cross-trajectory edges (KEY FIX) ──────────────────────────────
        # Connect nodes from DIFFERENT trajectories when spatially close.
        # This creates genuine routing choices: the planner can switch
        # between P000 (high risk) and P001 (low risk) corridors.
        # CoT-Route will prefer the lower-risk trajectory; Geometric A*
        # will pick the shortest regardless of risk.
        CROSS_TRAJ_DIST = MAX_EDGE_DIST_M * 3.0  # wider radius for cross-traj
        traj_names = list(traj_groups.keys())
        cross_edges = 0
        for ti in range(len(traj_names)):
            for tj in range(ti + 1, len(traj_names)):
                recs_i = traj_groups[traj_names[ti]]
                recs_j = traj_groups[traj_names[tj]]
                # Subsample to avoid O(n^2) explosion on long trajectories
                step_i = max(1, len(recs_i) // 30)
                step_j = max(1, len(recs_j) // 30)
                for ri in recs_i[::step_i]:
                    for rj in recs_j[::step_j]:
                        dist = float(np.linalg.norm(
                            np.array(ri["position"]) - np.array(rj["position"])))
                        if dist <= CROSS_TRAJ_DIST:
                            s    = (ri["semantic_risk"] + rj["semantic_risk"]) / 2
                            u    = (ri["uncertainty"]   + rj["uncertainty"])   / 2
                            cost = alpha * dist + beta * s + gamma * u
                            for a, b in [(ri["keyframe_idx"], rj["keyframe_idx"]),
                                         (rj["keyframe_idx"], ri["keyframe_idx"])]:
                                if not G.has_edge(a, b):
                                    G.add_edge(a, b, distance=round(dist,4),
                                               semantic_risk=round(s,4),
                                               uncertainty=round(u,4),
                                               cost=round(cost,4))
                                    cross_edges += 1
        log(f"  [{env}] Cross-trajectory edges added: {cross_edges}")

        graphs[env] = G
        graph_path  = GRAPHS_DIR / f"graph_{env}.json"
        with open(graph_path, "w") as f:
            json.dump(nx.node_link_data(G), f, indent=2)
        log(f"  [{env}] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Save primary graph (abandonedfactory or first available)
    primary_env = "abandonedfactory" if "abandonedfactory" in graphs \
                  else list(graphs.keys())[0]
    with open(GRAPHS_DIR / "navigation_graph.json", "w") as f:
        json.dump(nx.node_link_data(graphs[primary_env]), f, indent=2)
    log(f"Primary graph: {primary_env}")

    return graphs, primary_env

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Labels
# ══════════════════════════════════════════════════════════════════════════════
def generate_labels(all_records, graphs):
    log("=" * 62)
    log("STEP 4 — Generating training labels")
    log("=" * 62)

    alpha, beta, gamma = 0.4, 0.35, 0.25
    node_graph = {nid: G for G in graphs.values() for nid in G.nodes()}
    labels     = []

    for r in all_records:
        nid = r["keyframe_idx"]
        G   = node_graph.get(nid)
        if G and G.out_degree(nid) > 0:
            gt_cost = float(np.mean([G[nid][nb]["cost"] for nb in G.successors(nid)]))
        else:
            gt_cost = alpha*0.15*10 + beta*r["semantic_risk"] + gamma*r["uncertainty"]

        labels.append({
            "keyframe_idx":      nid,
            "frame_idx":         r["frame_idx"],
            "env":               r["env"],
            "traj":              r["traj"],
            "position":          r["position"],
            "imu_text":          r["imu_text"],
            "rgb_path":          r["rgb_path"],
            "semantic_risk_gt":  r["semantic_risk"],
            "uncertainty_gt":    r["uncertainty"],
            "composite_cost_gt": round(gt_cost, 4),
            "label_source":      "automatic_depth_texture",
        })

    with open(LABELS_DIR / "training_labels.json", "w") as f:
        json.dump(labels, f, indent=2)

    risks   = [l["semantic_risk_gt"]  for l in labels]
    uncerts = [l["uncertainty_gt"]    for l in labels]
    log(f"Labels: {len(labels)}  "
        f"Risk mean={np.mean(risks):.3f} std={np.std(risks):.3f}  "
        f"Unc mean={np.mean(uncerts):.3f} std={np.std(uncerts):.3f}")
    return labels

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — Visualisations
# ══════════════════════════════════════════════════════════════════════════════
def visualise(all_records, graphs, labels, primary_env):
    log("=" * 62)
    log("STEP 5 — Generating figures")
    log("=" * 62)

    risks   = np.array([r["semantic_risk"] for r in all_records])
    uncerts = np.array([r["uncertainty"]   for r in all_records])
    costs   = np.array([l["composite_cost_gt"] for l in labels])

    # Label distributions
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, vals, color, title, xlabel in [
        (axes[0], risks,   "#E74C3C", "Semantic Risk Distribution",      "Risk Score"),
        (axes[1], uncerts, "#3498DB", "Uncertainty Distribution",         "Uncertainty Score"),
        (axes[2], costs,   "#27AE60", "Composite Cost C(v) Distribution", "C(v)"),
    ]:
        ax.hist(vals, bins=40, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(np.mean(vals), color="black", linestyle="--",
                   label=f"Mean: {np.mean(vals):.3f}")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"CoT-Route Phase 1 — Label Statistics ({len(all_records)} keyframes)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase1_label_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Navigation graph (primary env)
    if primary_env in graphs:
        G = graphs[primary_env]
        ns = sorted(G.nodes())
        pos= np.array([G.nodes[n]["position"] for n in ns])
        gr = np.array([G.nodes[n]["risk"]        for n in ns])
        gu = np.array([G.nodes[n]["uncertainty"] for n in ns])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, vals, cmap, label, title in [
            (axes[0], gr, "RdYlGn_r", "Semantic Risk",
             f"Navigation Graph — Semantic Risk ({primary_env})"),
            (axes[1], gu, "coolwarm",  "Uncertainty",
             f"Navigation Graph — Uncertainty ({primary_env})"),
        ]:
            sc = ax.scatter(pos[:, 0], pos[:, 1], c=vals, cmap=cmap,
                            vmin=0, vmax=max(vals.max(), 0.01),
                            s=15, zorder=3, alpha=0.7)
            for u, v in list(G.edges())[:3000]:
                p1 = G.nodes[u]["position"]
                p2 = G.nodes[v]["position"]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        "gray", alpha=0.08, linewidth=0.4)
            plt.colorbar(sc, ax=ax, label=label)
            ax.set_title(title, fontweight="bold", fontsize=10)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, alpha=0.3)
        plt.suptitle("CoT-Route Phase 1 — Navigation Graph",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "phase1_navigation_graph.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    # Sample frames (one per environment)
    envs_present = list(set(r["env"] for r in all_records))[:6]
    sample_recs  = []
    for env in envs_present:
        env_recs = [r for r in all_records if r["env"] == env]
        if env_recs:
            sample_recs.append(env_recs[len(env_recs) // 2])

    if sample_recs:
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        for ax, rec in zip(axes.flatten(), sample_recs):
            img = cv2.imread(rec["rgb_path"])
            if img is not None:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            color = "red" if rec["semantic_risk"] > 0.5 else \
                    "orange" if rec["semantic_risk"] > 0.25 else "green"
            ax.set_title(
                f"{rec['env']}  Risk:{rec['semantic_risk']:.3f}  "
                f"Unc:{rec['uncertainty']:.3f}",
                fontsize=8, color=color, fontweight="bold")
            ax.axis("off")
        for j in range(len(sample_recs), 6):
            axes.flatten()[j].set_visible(False)
        plt.suptitle("CoT-Route Phase 1 — Sample Frames Across Environments",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "phase1_sample_frames.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    log("All figures saved")

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    start = time.time()
    log("=" * 62)
    log("CoT-Route — Phase 1: Real TartanAir Data Pipeline")
    log(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Project : {BASE_DIR}")
    log("=" * 62)

    train_seqs, val_seqs = discover_all_sequences()
    if not train_seqs and not val_seqs:
        log("No TartanAir sequences found — check download", "ERROR")
        sys.exit(1)

    all_records, train_records, val_records = process_all_sequences(
        train_seqs, val_seqs)

    if not all_records:
        log("No records produced", "ERROR")
        sys.exit(1)

    graphs, primary_env = build_all_graphs(all_records)
    labels = generate_labels(all_records, graphs)
    visualise(all_records, graphs, labels, primary_env)

    elapsed = time.time() - start
    summary = {
        "phase":            "Phase 1 — Real TartanAir Data Pipeline",
        "completed_at":     datetime.now().isoformat(),
        "data_source":      "TartanAir (real)",
        "train_sequences":  len(train_seqs),
        "val_sequences":    len(val_seqs),
        "total_keyframes":  len(all_records),
        "train_keyframes":  len(train_records),
        "val_keyframes":    len(val_records),
        "environments":     list(graphs.keys()),
        "primary_env":      primary_env,
        "training_labels":  len(labels),
        "cost_weights":     {"alpha": 0.4, "beta": 0.35, "gamma": 0.25},
        "ready_for_phase2": True,
        "total_time_s":     round(elapsed, 1),
        "output_files": {
            "keyframe_records": str(PROCESSED_DIR / "keyframe_records.json"),
            "navigation_graph": str(GRAPHS_DIR    / "navigation_graph.json"),
            "training_labels":  str(LABELS_DIR    / "training_labels.json"),
        }
    }
    with open(BASE_DIR / "phase1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log("")
    log("=" * 62)
    log("PHASE 1 COMPLETE")
    log("=" * 62)
    log(f"Environments    : {len(graphs)}")
    log(f"Total keyframes : {len(all_records)}")
    log(f"Training labels : {len(labels)}")
    log(f"Total time      : {elapsed:.1f}s")
    log("")
    log("NEXT STEP: python phase2_vlm_reasoning.py")
    save_log()

if __name__ == "__main__":
    main()
    if sys.platform == "win32":
        input("\nPress Enter to close...")
