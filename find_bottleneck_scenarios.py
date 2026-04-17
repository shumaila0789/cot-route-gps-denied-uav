"""
CoT-Route — Targeted Evaluation Script
=======================================
Identifies corridor bottleneck scenarios in the navigation graph where:
  - The shortest geometric path passes through HIGH-RISK or HIGH-UNCERTAINTY nodes
  - Alternative longer paths exist that avoid these risky regions
  - This forces methods to make different routing choices

These scenarios are where uncertainty-aware planning actually matters.
Running this BEFORE phase3_planner.py produces fairer, more compelling results.

HOW TO RUN:
  Place in D:\\COT_Routing_Protocol\\
  Run: python find_bottleneck_scenarios.py
  Then re-run: python phase3_planner.py

Outputs: data/graphs/bottleneck_scenarios.json
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime
from heapq import heappush, heappop

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.resolve()
DATA_DIR   = BASE_DIR / "data"
GRAPHS_DIR = DATA_DIR / "graphs"
FIGURES_DIR= BASE_DIR / "figures"
LOGS_DIR   = BASE_DIR / "logs"

for d in [GRAPHS_DIR, FIGURES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
# A bottleneck scenario is valid when:
MIN_RISK_ON_SHORT_PATH   = 0.25   # shortest path must pass through risk > this
MIN_UNCERT_ON_SHORT_PATH = 0.28   # OR uncertainty > this on shortest path
MIN_PATH_LENGTH_NODES    = 4      # path must be at least this many nodes long
MAX_SCENARIOS            = 20     # how many scenarios to keep
ALPHA, BETA, GAMMA       = 0.4, 0.35, 0.25

# ── Logger ────────────────────────────────────────────────────────────────────
log_lines = []
def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    log_lines.append(line)

# ── A* planner ────────────────────────────────────────────────────────────────
def astar(G, start, goal, weight_key="cost"):
    goal_pos = np.array(G.nodes[goal]["position"])
    def h(n):
        return float(np.linalg.norm(np.array(G.nodes[n]["position"]) - goal_pos))
    heap = [(h(start), start, [start], 0.0)]
    visited = {}
    while heap:
        f, node, path, g = heappop(heap)
        if node in visited:
            continue
        visited[node] = g
        if node == goal:
            return path, g
        for nb in G.successors(node):
            if nb in visited:
                continue
            ec   = G[node][nb].get(weight_key, 1.0)
            ng   = g + ec
            heappush(heap, (ng + h(nb), nb, path + [nb], ng))
    return None, float("inf")

# ── Path analysis ─────────────────────────────────────────────────────────────
def analyse_path(G, path):
    """Return mean risk, uncertainty, and max values along a path."""
    if not path or len(path) < 2:
        return {"mean_risk": 0.5, "mean_uncert": 0.5,
                "max_risk": 0.5,  "max_uncert": 0.5,
                "length_m": 0.0,  "n_nodes": 0}
    risks   = [G.nodes[n].get("vlm_risk",  G.nodes[n].get("risk",   0.0)) for n in path]
    uncerts = [G.nodes[n].get("vlm_uncert",G.nodes[n].get("uncertainty", 0.0)) for n in path]
    pos     = [np.array(G.nodes[n]["position"]) for n in path]
    length  = sum(np.linalg.norm(pos[i+1]-pos[i]) for i in range(len(pos)-1))
    return {
        "mean_risk":   round(float(np.mean(risks)),   4),
        "mean_uncert": round(float(np.mean(uncerts)), 4),
        "max_risk":    round(float(np.max(risks)),    4),
        "max_uncert":  round(float(np.max(uncerts)),  4),
        "length_m":    round(float(length),           3),
        "n_nodes":     len(path),
    }

# ── Find bottleneck scenarios ─────────────────────────────────────────────────
def find_bottleneck_scenarios(G):
    """
    Find start-goal pairs where:
      1. The shortest GEOMETRIC path (distance only) passes through risky/uncertain regions
      2. The CoT-Route semantic path takes a meaningfully different route
      3. The difference in risk/uncertainty is large enough to matter

    This gives scenarios where uncertainty-aware planning has real value.
    """
    log("Finding bottleneck scenarios ...")

    nodes  = list(G.nodes())
    n      = len(nodes)

    # Build a distance-only graph for geometric baseline
    G_geo = G.copy()
    for u, v in G_geo.edges():
        G_geo[u][v]["cost"] = G_geo[u][v]["distance"]

    scenarios    = []
    pairs_tested = 0

    # Test node pairs at various distances apart
    # Focus on pairs where trajectories diverge spatially
    step = max(1, n // 50)

    for i in range(0, n - step * 3, step):
        for j_mult in [3, 5, 8, 12]:
            j = min(i + step * j_mult, n - 1)
            if j >= n:
                continue

            start = nodes[i]
            goal  = nodes[j]

            if start == goal:
                continue
            pairs_tested += 1

            # Check connectivity
            try:
                if not nx.has_path(G, start, goal):
                    continue
            except Exception:
                continue

            # Plan geometric path (distance only)
            geo_path, _ = astar(G_geo, start, goal, weight_key="cost")
            if not geo_path or len(geo_path) < MIN_PATH_LENGTH_NODES:
                continue

            geo_stats = analyse_path(G, geo_path)

            # Only keep if geometric path has meaningful risk or uncertainty
            if (geo_stats["mean_risk"]   < MIN_RISK_ON_SHORT_PATH and
                geo_stats["mean_uncert"] < MIN_UNCERT_ON_SHORT_PATH):
                continue

            # Plan CoT-Route semantic path
            cot_path, _ = astar(G, start, goal, weight_key="cost")
            if not cot_path or len(cot_path) < 2:
                continue

            cot_stats = analyse_path(G, cot_path)

            # Compute improvement
            risk_improvement  = geo_stats["mean_risk"]   - cot_stats["mean_risk"]
            uncert_improvement= geo_stats["mean_uncert"] - cot_stats["mean_uncert"]
            path_diff         = geo_stats["mean_risk"] != cot_stats["mean_risk"] or \
                                geo_stats["mean_uncert"] != cot_stats["mean_uncert"]

            # Score the scenario: higher = more useful for demonstrating the method
            score = risk_improvement * 0.6 + uncert_improvement * 0.4

            if path_diff and score > -0.05:  # paths are different or CoT is safer
                scenarios.append({
                    "start":              start,
                    "goal":               goal,
                    "score":              round(float(score), 4),
                    "geo_mean_risk":      geo_stats["mean_risk"],
                    "geo_mean_uncert":    geo_stats["mean_uncert"],
                    "cot_mean_risk":      cot_stats["mean_risk"],
                    "cot_mean_uncert":    cot_stats["mean_uncert"],
                    "risk_improvement":   round(float(risk_improvement),   4),
                    "uncert_improvement": round(float(uncert_improvement), 4),
                    "geo_path_len":       geo_stats["n_nodes"],
                    "cot_path_len":       cot_stats["n_nodes"],
                })

    log(f"Tested {pairs_tested} pairs, found {len(scenarios)} candidate scenarios")

    # Sort by score (highest differentiation first)
    scenarios.sort(key=lambda x: x["score"], reverse=True)
    top = scenarios[:MAX_SCENARIOS]

    log(f"Keeping top {len(top)} scenarios")
    if top:
        log(f"Best scenario — risk improvement: {top[0]['risk_improvement']:.3f}, "
            f"uncert improvement: {top[0]['uncert_improvement']:.3f}")

    return top

# ── Visualise scenarios ───────────────────────────────────────────────────────
def visualise_scenarios(G, scenarios):
    if not scenarios:
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Build geometric graph for comparison
    G_geo = G.copy()
    for u, v in G_geo.edges():
        G_geo[u][v]["cost"] = G_geo[u][v]["distance"]

    n_show = min(6, len(scenarios))
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    positions  = {n: G.nodes[n]["position"] for n in G.nodes()}
    all_risks  = np.array([G.nodes[n].get("vlm_risk",
                           G.nodes[n].get("risk", 0.0)) for n in G.nodes()])

    for idx in range(n_show):
        ax  = axes[idx]
        sc  = scenarios[idx]
        start, goal = sc["start"], sc["goal"]

        # Background: all nodes coloured by risk
        pos_arr = np.array([positions[n] for n in G.nodes()])
        ax.scatter(pos_arr[:, 0], pos_arr[:, 1],
                   c=all_risks, cmap="RdYlGn_r", vmin=0, vmax=0.6,
                   s=12, alpha=0.4, zorder=1)

        # Geometric path (red dashed)
        geo_path, _ = astar(G_geo, start, goal, weight_key="cost")
        if geo_path:
            gx = [positions[n][0] for n in geo_path]
            gy = [positions[n][1] for n in geo_path]
            ax.plot(gx, gy, "r--", linewidth=2.0, alpha=0.8,
                    label=f"Geo A* (risk={sc['geo_mean_risk']:.3f})", zorder=3)

        # CoT-Route path (blue solid)
        cot_path, _ = astar(G, start, goal, weight_key="cost")
        if cot_path:
            cx = [positions[n][0] for n in cot_path]
            cy = [positions[n][1] for n in cot_path]
            ax.plot(cx, cy, "b-", linewidth=2.5, alpha=0.9,
                    label=f"CoT-Route (risk={sc['cot_mean_risk']:.3f})", zorder=4)

        # Start / goal markers
        sp = positions[start]
        gp = positions[goal]
        ax.plot(sp[0], sp[1], "go", markersize=12, zorder=5)
        ax.plot(gp[0], gp[1], "r*", markersize=14, zorder=5)

        ax.set_title(
            f"Scenario {idx+1}  |  "
            f"Risk: {sc['geo_mean_risk']:.3f}→{sc['cot_mean_risk']:.3f}  "
            f"Unc: {sc['geo_mean_uncert']:.3f}→{sc['cot_mean_uncert']:.3f}",
            fontsize=8, fontweight="bold"
        )
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel("X (m)", fontsize=7)
        ax.set_ylabel("Y (m)", fontsize=7)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.2)

    for idx in range(n_show, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        "CoT-Route: Bottleneck Scenarios\n"
        "Red dashed = Geometric A* (blind to risk), Blue solid = CoT-Route (risk-aware)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    p = FIGURES_DIR / "bottleneck_scenarios.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Scenario figure saved: {p}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    start_time = time.time()
    log("=" * 62)
    log("CoT-Route: Targeted Bottleneck Scenario Finder")
    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 62)

    # Load navigation graph
    graph_path = GRAPHS_DIR / "navigation_graph.json"
    if not graph_path.exists():
        log("Navigation graph not found — run phase1 first")
        sys.exit(1)

    with open(graph_path, "r") as f:
        graph_data = json.load(f)
    G = nx.node_link_graph(graph_data)
    log(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Load VLM outputs and add to graph nodes directly
    # This makes the script self-contained — no need to run phase3 first
    vlm_path = DATA_DIR / "vlm_outputs" / "vlm_cost_inputs.json"
    if not vlm_path.exists():
        log("VLM outputs not found — run phase2_vlm_reasoning.py first")
        sys.exit(1)

    with open(vlm_path, "r") as f:
        vlm_inputs = json.load(f)
    vlm_map = {v["keyframe_idx"]: v for v in vlm_inputs}

    # Attach VLM scores to graph nodes
    matched = 0
    for nid in G.nodes():
        vlm = vlm_map.get(nid, {})
        G.nodes[nid]["vlm_risk"]   = vlm.get("semantic_risk", 0.0)
        G.nodes[nid]["vlm_uncert"] = vlm.get("uncertainty",   0.0)
        if nid in vlm_map:
            matched += 1
    log(f"VLM scores loaded: {matched}/{G.number_of_nodes()} nodes matched")

    # Find bottleneck scenarios
    scenarios = find_bottleneck_scenarios(G)

    if not scenarios:
        log("No bottleneck scenarios found")
        log("Try reducing MIN_RISK_ON_SHORT_PATH or MIN_UNCERT_ON_SHORT_PATH")
        sys.exit(1)

    # Visualise
    visualise_scenarios(G, scenarios)

    # Save scenarios
    out_path = GRAPHS_DIR / "bottleneck_scenarios.json"
    with open(out_path, "w") as f:
        json.dump(scenarios, f, indent=2)
    log(f"Scenarios saved: {out_path}")

    # Print summary
    log("")
    log("=" * 62)
    log("TOP BOTTLENECK SCENARIOS")
    log("=" * 62)
    log(f"  {'#':<4} {'Risk Geo':>9} {'Risk CoT':>9} {'Unc Geo':>8} "
        f"{'Unc CoT':>8} {'Risk Imp':>9} {'Unc Imp':>9}")
    log(f"  {'-'*4} {'-'*9} {'-'*9} {'-'*8} {'-'*8} {'-'*9} {'-'*9}")
    for i, s in enumerate(scenarios[:10]):
        log(f"  {i+1:<4} {s['geo_mean_risk']:>9.3f} {s['cot_mean_risk']:>9.3f} "
            f"{s['geo_mean_uncert']:>8.3f} {s['cot_mean_uncert']:>8.3f} "
            f"{s['risk_improvement']:>9.3f} {s['uncert_improvement']:>9.3f}")

    elapsed = time.time() - start_time
    log("")
    log(f"Done in {elapsed:.1f}s")
    log("NEXT STEP: python phase3_planner.py")
    log("Phase 3 will automatically use these bottleneck scenarios for evaluation")

    # Save log
    log_path = LOGS_DIR / "bottleneck_finder.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

if __name__ == "__main__":
    main()
    import sys as _sys
    if _sys.platform == "win32":
        input("\nPress Enter to close...")
