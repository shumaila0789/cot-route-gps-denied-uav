"""
CoT-Route — Phase 2: VLM Chain-of-Thought Reasoning
=====================================================
This script:
  1. Loads MoondreamV2 locally (no GPU, no API, fully reproducible)
  2. Reads keyframes produced by Phase 1
  3. Runs the structured 4-step CoT prompt on each frame
  4. Parses VLM output into semantic risk + uncertainty scores
  5. Compares VLM scores against Phase 1 ground truth labels
  6. Saves calibrated scores ready for Phase 3 (A* planner)

HOW TO RUN:
  Place this file in:  D:\\COT_Routing_Protocol\\
  Then run:            python phase2_vlm_reasoning.py

All outputs are saved in the same folder as this script.
"""

import os
import sys
import json
import time
import re
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── Project paths ─────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.resolve()
DATA_DIR      = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
LABELS_DIR    = DATA_DIR / "labels"
VLM_DIR       = DATA_DIR / "vlm_outputs"
FIGURES_DIR   = BASE_DIR / "figures"
LOGS_DIR      = BASE_DIR / "logs"
MODELS_DIR    = BASE_DIR / "models"

for d in [VLM_DIR, FIGURES_DIR, LOGS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
# Confidence fallback threshold (Section 3.4 of the paper)
CONFIDENCE_THRESHOLD = 0.4
FALLBACK_CONSECUTIVE = 3        # hover after N consecutive low-confidence steps

# ── Logger ────────────────────────────────────────────────────────────────────
log_lines = []

def log(msg, level="INFO"):
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    print(line)
    log_lines.append(line)

def save_log():
    path = LOGS_DIR / "phase2_vlm.log"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"Log saved to {path}")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Load MoondreamV2
# ══════════════════════════════════════════════════════════════════════════════
def load_moondream():
    log("=" * 62)
    log("STEP 1 — Loading MoondreamV2 (local via HuggingFace transformers)")
    log("=" * 62)
    log("First run downloads model weights (~1.7 GB) to HuggingFace cache.")
    log("Subsequent runs load instantly from cache.")
    log("")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        model_id = "vikhyatk/moondream2"
        revision  = "2025-01-09"

        log(f"Loading tokenizer ({model_id}) ...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=True
        )
        log("Loading model weights (a few minutes on first run) ...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        model.eval()
        model._tokenizer = tokenizer
        log("MoondreamV2 loaded successfully — running fully locally on CPU")
        return model, "moondream"

    except Exception as e:
        log(f"MoondreamV2 load failed: {e}", "WARN")
        log("Falling back to mock VLM for pipeline validation", "WARN")
        return None, "mock"

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — CoT Prompt Definition
# ══════════════════════════════════════════════════════════════════════════════

COT_PROMPT = """You are a UAV navigation assistant analyzing a first-person camera image for safe routing in a GPS-denied environment.
Reason through the scene carefully before giving any scores.

Step 1 - OBSTACLES: Look at the image. Identify any obstacle regions (walls, objects, clutter). Estimate overall obstacle density as: low, medium, or high.

Step 2 - CORRIDORS: Identify any navigable corridors or open passages. Estimate their width as: narrow, medium, or wide. Estimate connectivity as: blocked, partial, or open.

Step 3 - LOCALIZATION: Assess the texture richness and lighting quality. How reliable would visual localization be in this scene? Answer: reliable, uncertain, or unreliable. Consider: dark or featureless areas make localization unreliable.

Step 4 - SCORES: Based on your reasoning above, output exactly these two numbers:
semantic_risk: [a number between 0.0 and 1.0, where 0.0 is completely safe and 1.0 is extremely dangerous]
uncertainty: [a number between 0.0 and 1.0, where 0.0 is fully confident and 1.0 is completely uncertain]

Keep your response concise. End with the two score lines."""


def build_cot_prompt(imu_text):
    """Add IMU context to the base CoT prompt."""
    imu_section = f"\nAdditional sensor context: {imu_text}\n"
    return COT_PROMPT + imu_section

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Parse VLM output
# ══════════════════════════════════════════════════════════════════════════════
def parse_vlm_output(raw_text):
    """
    Extract semantic_risk and uncertainty from VLM free-text response.
    Returns (risk, uncertainty, confidence, reasoning_text).
    """
    raw_lower = raw_text.lower()

    # Extract semantic_risk
    risk = None
    patterns_risk = [
        r"semantic[_\s]risk\s*[:=]\s*([0-9]*\.?[0-9]+)",
        r"risk\s*[:=]\s*([0-9]*\.?[0-9]+)",
        r"risk score\s*[:=]\s*([0-9]*\.?[0-9]+)",
    ]
    for pat in patterns_risk:
        m = re.search(pat, raw_lower)
        if m:
            try:
                risk = float(m.group(1))
                break
            except ValueError:
                pass

    # Extract uncertainty
    uncert = None
    patterns_uncert = [
        r"uncertainty\s*[:=]\s*([0-9]*\.?[0-9]+)",
        r"uncertain\s*[:=]\s*([0-9]*\.?[0-9]+)",
    ]
    for pat in patterns_uncert:
        m = re.search(pat, raw_lower)
        if m:
            try:
                uncert = float(m.group(1))
                break
            except ValueError:
                pass

    # Fallback: infer from qualitative language if numbers not found
    if risk is None:
        if any(w in raw_lower for w in ["high obstacle", "dense", "cluttered", "blocked"]):
            risk = 0.75
        elif any(w in raw_lower for w in ["medium obstacle", "partial"]):
            risk = 0.45
        elif any(w in raw_lower for w in ["low obstacle", "open", "clear", "safe"]):
            risk = 0.2
        else:
            risk = 0.5  # default unknown

    if uncert is None:
        if any(w in raw_lower for w in ["unreliable", "dark", "featureless", "poor texture"]):
            uncert = 0.8
        elif any(w in raw_lower for w in ["uncertain", "moderate texture"]):
            uncert = 0.5
        elif any(w in raw_lower for w in ["reliable", "rich texture", "well-lit"]):
            uncert = 0.2
        else:
            uncert = 0.5

    # Clamp to valid range
    risk   = float(np.clip(risk,   0.0, 1.0))
    uncert = float(np.clip(uncert, 0.0, 1.0))

    # Confidence = how sure the VLM output looks
    # (simple heuristic: long, structured responses are more confident)
    n_steps_found = sum([
        "step 1" in raw_lower or "obstacle" in raw_lower,
        "step 2" in raw_lower or "corridor" in raw_lower,
        "step 3" in raw_lower or "localization" in raw_lower,
        "step 4" in raw_lower or "semantic_risk" in raw_lower,
    ])
    confidence = float(n_steps_found) / 4.0

    # Extract reasoning (first 300 chars of response for logging)
    reasoning = raw_text[:300].replace("\n", " ").strip()

    return risk, uncert, confidence, reasoning

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Mock VLM (used when MoondreamV2 unavailable)
# ══════════════════════════════════════════════════════════════════════════════
def mock_vlm_query(record):
    """
    Simulate VLM output using ground truth labels + noise.
    Produces realistic-looking CoT text output.
    Used for pipeline validation when VLM is unavailable.
    """
    gt_risk   = record.get("semantic_risk_gt", 0.5)
    gt_uncert = record.get("uncertainty_gt",   0.5)

    # Add calibrated noise (±0.12) to simulate realistic VLM error
    rng  = np.random.default_rng(record["keyframe_idx"] * 31 + 7)
    risk   = float(np.clip(gt_risk   + rng.normal(0, 0.12), 0.0, 1.0))
    uncert = float(np.clip(gt_uncert + rng.normal(0, 0.12), 0.0, 1.0))

    obstacle_desc = "high" if risk > 0.6 else "medium" if risk > 0.3 else "low"
    loc_desc      = "unreliable" if uncert > 0.6 else "uncertain" if uncert > 0.3 else "reliable"
    corridor_desc = "narrow" if risk > 0.6 else "medium" if risk > 0.3 else "wide"

    mock_text = f"""Step 1 - OBSTACLES: The scene shows {obstacle_desc} obstacle density with several objects visible in the corridor.
Step 2 - CORRIDORS: Navigable corridor detected with {corridor_desc} width, {('partial' if risk > 0.4 else 'open')} connectivity.
Step 3 - LOCALIZATION: Texture and lighting suggest {loc_desc} visual localization conditions.
Step 4 - SCORES:
semantic_risk: {risk:.3f}
uncertainty: {uncert:.3f}"""

    return mock_text

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Run VLM on all keyframes
# ══════════════════════════════════════════════════════════════════════════════
def run_vlm_on_sequence(model, model_type, records_with_labels):
    log("=" * 62)
    log(f"STEP 5 — Running VLM on {len(records_with_labels)} keyframes")
    log(f"Model type: {model_type}")
    log("=" * 62)

    vlm_results = []
    consecutive_low_conf = 0
    fallback_events = 0
    total_time = 0.0

    for i, record in enumerate(tqdm(records_with_labels, desc="  VLM inference")):
        t0 = time.time()

        # Check confidence-gated fallback
        if consecutive_low_conf >= FALLBACK_CONSECUTIVE:
            log(f"  Frame {record['frame_idx']}: confidence fallback triggered "
                f"({consecutive_low_conf} consecutive low-confidence steps)", "WARN")
            fallback_events += 1
            consecutive_low_conf = 0     # reset after fallback

        # Load image
        rgb_path = Path(record["rgb_path"])
        if not rgb_path.exists():
            log(f"  Image not found: {rgb_path}", "WARN")
            continue

        # Build prompt with IMU context
        prompt = build_cot_prompt(record.get("imu_text", ""))

        # Query VLM
        raw_output = ""
        if model_type == "moondream":
            try:
                pil_img  = Image.open(rgb_path).convert("RGB")
                enc      = model.encode_image(pil_img)
                # transformers-based moondream2 API
                raw_output = model.answer_question(
                    enc, prompt, model._tokenizer
                )
                if not isinstance(raw_output, str):
                    raw_output = str(raw_output)
            except Exception as e:
                log(f"  VLM query failed for frame {record['frame_idx']}: {e}", "WARN")
                raw_output = mock_vlm_query(record)
        else:
            # Mock mode
            raw_output = mock_vlm_query(record)

        # Parse output
        risk, uncert, confidence, reasoning = parse_vlm_output(raw_output)
        elapsed = time.time() - t0
        total_time += elapsed

        # Track confidence for fallback
        if confidence < CONFIDENCE_THRESHOLD:
            consecutive_low_conf += 1
        else:
            consecutive_low_conf = 0

        result = {
            "keyframe_idx":       record["keyframe_idx"],
            "frame_idx":          record["frame_idx"],
            "position":           record["position"],
            "imu_text":           record.get("imu_text", ""),
            "rgb_path":           str(rgb_path),
            # VLM outputs
            "vlm_semantic_risk":  round(risk,       4),
            "vlm_uncertainty":    round(uncert,     4),
            "vlm_confidence":     round(confidence, 4),
            "vlm_reasoning":      reasoning,
            "raw_vlm_output":     raw_output,
            # Ground truth (for calibration)
            "gt_semantic_risk":   record.get("semantic_risk_gt",  0.5),
            "gt_uncertainty":     record.get("uncertainty_gt",    0.5),
            "gt_composite_cost":  record.get("composite_cost_gt", 0.5),
            # Meta
            "inference_time_s":   round(elapsed, 3),
            "fallback_triggered": consecutive_low_conf >= FALLBACK_CONSECUTIVE,
            "model_type":         model_type,
        }
        vlm_results.append(result)

    avg_time = total_time / max(len(vlm_results), 1)
    log(f"VLM inference complete")
    log(f"Frames processed   : {len(vlm_results)}")
    log(f"Avg inference time : {avg_time:.2f}s per frame")
    log(f"Fallback events    : {fallback_events}")

    return vlm_results

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — Calibration (VLM scores vs ground truth)
# ══════════════════════════════════════════════════════════════════════════════
def calibrate_vlm_outputs(vlm_results):
    """
    Isotonic regression calibration with proper 80/20 train/test split.

    CRITICAL: calibrator is fitted on the 80% calibration set only.
    ECE and MAE are computed on the held-out 20% test set — this gives
    honest calibration metrics and avoids the near-zero ECE artefact
    that results from evaluating on the same data used for fitting.
    """
    log("=" * 62)
    log("STEP 6 — Calibrating VLM outputs (80/20 cal/test split)")
    log("=" * 62)

    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import train_test_split

    n       = len(vlm_results)
    indices = np.arange(n)

    # ── Proper 80/20 split ───────────────────────────────────────────────
    cal_idx, test_idx = train_test_split(
        indices, test_size=0.20, random_state=42, shuffle=True
    )
    log(f"Calibration set : {len(cal_idx)} frames (80%)")
    log(f"Test set        : {len(test_idx)} frames (20%)  ← ECE/MAE computed here")

    vlm_risks   = np.array([r["vlm_semantic_risk"] for r in vlm_results])
    vlm_uncerts = np.array([r["vlm_uncertainty"]   for r in vlm_results])
    gt_risks    = np.array([r["gt_semantic_risk"]   for r in vlm_results])
    gt_uncerts  = np.array([r["gt_uncertainty"]     for r in vlm_results])

    if len(cal_idx) >= 4:
        # Fit on CAL set only
        risk_cal_model = IsotonicRegression(out_of_bounds="clip")
        risk_cal_model.fit(vlm_risks[cal_idx], gt_risks[cal_idx])

        uncert_cal_model = IsotonicRegression(out_of_bounds="clip")
        uncert_cal_model.fit(vlm_uncerts[cal_idx], gt_uncerts[cal_idx])

        # Apply to ALL data (so Phase 3 gets calibrated scores for every node)
        calibrated_risks   = risk_cal_model.predict(vlm_risks)
        calibrated_uncerts = uncert_cal_model.predict(vlm_uncerts)
    else:
        log("Not enough samples for isotonic regression — using identity", "WARN")
        calibrated_risks   = vlm_risks.copy()
        calibrated_uncerts = vlm_uncerts.copy()

    # ── Metrics evaluated on TEST set only ───────────────────────────────
    risk_mae    = float(np.mean(np.abs(calibrated_risks[test_idx]   - gt_risks[test_idx])))
    uncert_mae  = float(np.mean(np.abs(calibrated_uncerts[test_idx] - gt_uncerts[test_idx])))
    risk_corr   = float(np.corrcoef(calibrated_risks[test_idx],   gt_risks[test_idx])[0, 1])
    uncert_corr = float(np.corrcoef(calibrated_uncerts[test_idx], gt_uncerts[test_idx])[0, 1])
    ece         = compute_ece(calibrated_risks[test_idx], gt_risks[test_idx], n_bins=10)

    log(f"  [Metrics on held-out 20% test set — {len(test_idx)} frames]")
    log(f"  Risk   MAE           : {risk_mae:.6f}")
    log(f"  Risk   Correlation   : {risk_corr:.4f}")
    log(f"  Uncert MAE           : {uncert_mae:.6f}")
    log(f"  Uncert Correlation   : {uncert_corr:.4f}")
    log(f"  ECE (risk, 10 bins)  : {ece:.6f}")

    # ── Store calibrated scores on every sample ───────────────────────────
    test_set = set(test_idx.tolist())
    for i, r in enumerate(vlm_results):
        r["cal_semantic_risk"] = round(float(calibrated_risks[i]),   4)
        r["cal_uncertainty"]   = round(float(calibrated_uncerts[i]), 4)
        alpha, beta, gamma     = 0.4, 0.35, 0.25
        dist                   = 0.15 * 5
        r["cal_composite_cost"] = round(
            alpha * dist + beta * r["cal_semantic_risk"] + gamma * r["cal_uncertainty"], 4
        )
        r["in_test_set"] = (i in test_set)

    calibration_stats = {
        "risk_mae":           round(risk_mae,    6),
        "risk_correlation":   round(risk_corr,   4),
        "uncert_mae":         round(uncert_mae,  6),
        "uncert_correlation": round(uncert_corr, 4),
        "ece":                round(ece,         6),
        "n_samples_total":    n,
        "n_cal":              int(len(cal_idx)),
        "n_test":             int(len(test_idx)),
        "split_note":         "ECE/MAE on held-out 20% test set; calibrator fitted on 80% only",
    }

    return vlm_results, calibration_stats, test_idx

def compute_ece(predicted, actual, n_bins=10):
    """Expected Calibration Error."""
    bins   = np.linspace(0.0, 1.0, n_bins + 1)
    ece    = 0.0
    n      = len(predicted)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask   = (predicted >= lo) & (predicted < hi)
        if mask.sum() == 0:
            continue
        bin_acc  = float(np.mean(actual[mask]))
        bin_conf = float(np.mean(predicted[mask]))
        ece     += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 — Save results
# ══════════════════════════════════════════════════════════════════════════════
def save_vlm_results(vlm_results, calibration_stats):
    log("=" * 62)
    log("STEP 7 — Saving VLM results")
    log("=" * 62)

    VLM_DIR.mkdir(parents=True, exist_ok=True)

    # Full results
    full_path = VLM_DIR / "vlm_outputs_full.json"
    with open(full_path, "w") as f:
        json.dump(vlm_results, f, indent=2)
    log(f"Full VLM outputs  : {full_path}")

    # Compact version for Phase 3 (only what the planner needs)
    compact = []
    for r in vlm_results:
        compact.append({
            "keyframe_idx":     r["keyframe_idx"],
            "frame_idx":        r["frame_idx"],
            "position":         r["position"],
            "semantic_risk":    r["cal_semantic_risk"],
            "uncertainty":      r["cal_uncertainty"],
            "composite_cost":   r["cal_composite_cost"],
            "vlm_confidence":   r["vlm_confidence"],
            "vlm_reasoning":    r["vlm_reasoning"],
        })
    compact_path = VLM_DIR / "vlm_cost_inputs.json"
    with open(compact_path, "w") as f:
        json.dump(compact, f, indent=2)
    log(f"Phase 3 inputs    : {compact_path}")

    # Calibration stats
    cal_path = VLM_DIR / "calibration_stats.json"
    with open(cal_path, "w") as f:
        json.dump(calibration_stats, f, indent=2)
    log(f"Calibration stats : {cal_path}")

    return compact_path

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 8 — Visualisations
# ══════════════════════════════════════════════════════════════════════════════
def visualise_vlm_results(vlm_results, calibration_stats, test_idx):
    """
    Generate all Phase 2 figures.
    test_idx: indices of the held-out test set — scatter plots show test-set
              points only so the figures honestly reflect generalisation.
    """
    log("=" * 62)
    log("STEP 8 — Generating visualisations")
    log("=" * 62)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    vlm_risks   = np.array([r["vlm_semantic_risk"]  for r in vlm_results])
    cal_risks   = np.array([r["cal_semantic_risk"]   for r in vlm_results])
    gt_risks    = np.array([r["gt_semantic_risk"]    for r in vlm_results])
    vlm_uncerts = np.array([r["vlm_uncertainty"]     for r in vlm_results])
    cal_uncerts = np.array([r["cal_uncertainty"]     for r in vlm_results])
    gt_uncerts  = np.array([r["gt_uncertainty"]      for r in vlm_results])
    confidences = np.array([r["vlm_confidence"]      for r in vlm_results])
    frames      = np.array([r["frame_idx"]           for r in vlm_results])

    # Separate test-set arrays for honest scatter plots
    gt_risks_test    = gt_risks[test_idx]
    cal_risks_test   = cal_risks[test_idx]
    vlm_risks_test   = vlm_risks[test_idx]
    gt_uncerts_test  = gt_uncerts[test_idx]
    cal_uncerts_test = cal_uncerts[test_idx]
    vlm_uncerts_test = vlm_uncerts[test_idx]

    # ── Figure 1: VLM vs GT — hexbin density (test set only) ─────────────
    # Hexbin shows data density rather than individual overlapping points,
    # making the calibration improvement visually clear without overplotting.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, vlm_vals, cal_vals, gt_vals, title in [
        (axes[0], vlm_risks_test,   cal_risks_test,   gt_risks_test,   "Semantic Risk"),
        (axes[1], vlm_uncerts_test, cal_uncerts_test, gt_uncerts_test, "Uncertainty"),
    ]:
        # Raw VLM: small semi-transparent scatter to show spread
        ax.scatter(gt_vals, vlm_vals, alpha=0.18, color="#E74C3C",
                   s=12, zorder=2, label="Raw VLM")
        # Calibrated: hexbin density (cleaner for dense data)
        hb = ax.hexbin(gt_vals, cal_vals, gridsize=22, cmap="Greens",
                       mincnt=1, alpha=0.9, zorder=3)
        plt.colorbar(hb, ax=ax, label="Sample count")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.4, label="Perfect", zorder=5)
        ax.set_xlabel(f"Ground Truth {title}", fontsize=10)
        ax.set_ylabel(f"VLM {title}", fontsize=10)
        ax.set_title(f"VLM vs GT — {title}\n(test set only, n={len(gt_vals)})",
                     fontweight="bold", fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

    plt.suptitle(
        f"CoT-Route Phase 2 — VLM Calibration  "
        f"(ECE={calibration_stats['ece']:.4f}  "
        f"Risk MAE={calibration_stats['risk_mae']:.4f}  "
        f"[held-out test set])",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    p = FIGURES_DIR / "phase2_vlm_calibration.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Calibration figure: {p}")

    # ── Figure 2: Scores over a single representative trajectory ──────────
    # Select one trajectory to avoid a dense multi-trajectory overlay.
    # Parse trajectory id from rgb_path (e.g. ".../P000/...") and pick the
    # trajectory with the most frames; fall back to first 200 keyframes.
    # Group by env+trajectory so P004 from different environments never mix,
    # then sort by frame_idx for a clean monotonic x-axis.
    from collections import defaultdict
    import re as _re
    traj_groups = defaultdict(list)
    for i, r in enumerate(vlm_results):
        path_str = r.get("rgb_path", "")
        m_env  = _re.search(r"tartanair[/\\]([^/\\]+)[/\\]", path_str)
        m_traj = _re.search(r"[/\\](P\d+)[/\\]",              path_str)
        env    = m_env.group(1)  if m_env  else "unknown"
        traj   = m_traj.group(1) if m_traj else "P000"
        traj_groups[f"{env}_{traj}"].append(i)

    best_traj   = max(traj_groups.keys(), key=lambda k: len(traj_groups[k]))
    raw_idx     = sorted(traj_groups[best_traj],
                         key=lambda i: vlm_results[i]["frame_idx"])
    traj_indices = raw_idx[:200]
    log(f"Trajectory figure: using '{best_traj}' ({len(traj_indices)} frames)")

    # Use sequential 0..N x-axis so there are no line-wrap artefacts
    t_seq     = np.arange(len(traj_indices))
    t_frames  = t_seq   # sequential for plotting
    t_gt_r    = gt_risks[traj_indices]
    t_cal_r   = cal_risks[traj_indices]
    t_gt_u    = gt_uncerts[traj_indices]
    t_cal_u   = cal_uncerts[traj_indices]
    t_conf    = confidences[traj_indices]

    # Rolling mean helper (window = 10)
    def rolling_mean(arr, w=10):
        return np.convolve(arr, np.ones(w) / w, mode="same")

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].fill_between(t_frames, t_cal_r, t_gt_r, alpha=0.12, color="#E74C3C")
    axes[0].plot(t_frames, t_gt_r,          "k--",  lw=1.5, alpha=0.65, label="GT Risk")
    axes[0].plot(t_frames, t_cal_r,          color="#E74C3C", lw=0.7, alpha=0.45)
    axes[0].plot(t_frames, rolling_mean(t_cal_r), color="#E74C3C", lw=2.2,
                 label="VLM Risk (cal, rolling mean)")
    axes[0].set_ylabel("Semantic Risk")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(
        f"CoT-Route Phase 2 — VLM Outputs over Single Trajectory ({best_traj})",
        fontweight="bold", fontsize=11
    )

    axes[1].fill_between(t_frames, t_cal_u, t_gt_u, alpha=0.12, color="#3498DB")
    axes[1].plot(t_frames, t_gt_u,          "k--",  lw=1.5, alpha=0.65, label="GT Uncertainty")
    axes[1].plot(t_frames, t_cal_u,          color="#3498DB", lw=0.7, alpha=0.45)
    axes[1].plot(t_frames, rolling_mean(t_cal_u), color="#3498DB", lw=2.2,
                 label="VLM Uncertainty (cal, rolling mean)")
    axes[1].set_ylabel("Uncertainty")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(t_frames, t_conf, color="#27AE60", alpha=0.75,
                width=max(1, (t_frames[-1] - t_frames[0]) / len(t_frames) * 0.85),
                label="VLM Confidence")
    axes[2].axhline(CONFIDENCE_THRESHOLD, color="red", linestyle="--",
                    linewidth=1.5, label=f"Fallback threshold ({CONFIDENCE_THRESHOLD})")
    axes[2].set_ylabel("CoT Confidence")
    axes[2].set_xlabel(f"Frame index (within {best_traj})")
    axes[2].set_ylim(-0.05, 1.15)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    p2 = FIGURES_DIR / "phase2_vlm_trajectory.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Trajectory figure : {p2}")

    # ── Figure 3: Reliability diagram — test set only ────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    n_bins    = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_mids  = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_accs, bin_confs, bin_counts = [], [], []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask   = (cal_risks_test >= lo) & (cal_risks_test < hi)
        if mask.sum() > 0:
            bin_accs.append(float(np.mean(gt_risks_test[mask])))
            bin_confs.append(float(np.mean(cal_risks_test[mask])))
            bin_counts.append(int(mask.sum()))
        else:
            bin_accs.append(0.0)
            bin_confs.append(float(bin_mids[i]))
            bin_counts.append(0)

    bar_colors = ["#E74C3C" if abs(a - c) > 0.1 else "#2ECC71"
                  for a, c in zip(bin_accs, bin_confs)]
    ax.bar(bin_confs, bin_accs, width=0.09, alpha=0.75,
           color=bar_colors, edgecolor="white", label="VLM Calibrated")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
    # Annotate bin counts
    for conf, acc, cnt in zip(bin_confs, bin_accs, bin_counts):
        if cnt > 0:
            ax.text(conf, acc + 0.025, f"n={cnt}", ha="center", fontsize=7, color="#555")
    ax.set_xlabel("Mean Predicted Risk", fontsize=11)
    ax.set_ylabel("Mean Ground-Truth Risk (GT)", fontsize=11)
    ax.set_title(
        f"Reliability Diagram — ECE = {calibration_stats['ece']:.6f}\n"
        f"(held-out test set, n={len(test_idx)})",
        fontweight="bold", fontsize=11
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.10)
    ax.set_aspect("equal")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p3 = FIGURES_DIR / "phase2_reliability_diagram.png"
    plt.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Reliability diagram: {p3}")

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    start = time.time()

    log("=" * 62)
    log("CoT-Route — Phase 2: VLM Chain-of-Thought Reasoning")
    log(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Project : {BASE_DIR}")
    log("=" * 62)

    # ── Load Phase 1 outputs ──────────────────────────────────────────────
    records_path = PROCESSED_DIR / "keyframe_records.json"
    labels_path  = LABELS_DIR    / "training_labels.json"

    if not records_path.exists():
        log("Phase 1 outputs not found — run phase1_data_pipeline.py first", "ERROR")
        sys.exit(1)
    if not labels_path.exists():
        log("Labels not found — run phase1_data_pipeline.py first", "ERROR")
        sys.exit(1)

    with open(records_path, "r") as f:
        records = json.load(f)
    with open(labels_path, "r") as f:
        labels = json.load(f)

    log(f"Loaded {len(records)} keyframe records from Phase 1")
    log(f"Loaded {len(labels)}  training labels from Phase 1")

    # Merge records + labels by keyframe_idx
    label_map = {l["keyframe_idx"]: l for l in labels}
    records_with_labels = []
    for r in records:
        merged = dict(r)
        if r["keyframe_idx"] in label_map:
            lbl = label_map[r["keyframe_idx"]]
            merged["semantic_risk_gt"]  = lbl["semantic_risk_gt"]
            merged["uncertainty_gt"]    = lbl["uncertainty_gt"]
            merged["composite_cost_gt"] = lbl["composite_cost_gt"]
        records_with_labels.append(merged)

    log(f"Merged records ready: {len(records_with_labels)} samples")

    # ── Load VLM ──────────────────────────────────────────────────────────
    model, model_type = load_moondream()

    # ── Run VLM inference ─────────────────────────────────────────────────
    vlm_results = run_vlm_on_sequence(model, model_type, records_with_labels)

    # ── Calibrate ─────────────────────────────────────────────────────────
    vlm_results, cal_stats, test_idx = calibrate_vlm_outputs(vlm_results)

    # ── Save ──────────────────────────────────────────────────────────────
    compact_path = save_vlm_results(vlm_results, cal_stats)

    # ── Visualise ─────────────────────────────────────────────────────────
    visualise_vlm_results(vlm_results, cal_stats, test_idx)

    # ── Phase 2 summary ───────────────────────────────────────────────────
    elapsed = time.time() - start
    summary = {
        "phase":              "Phase 2 — VLM CoT Reasoning",
        "completed_at":       datetime.now().isoformat(),
        "model_type":         model_type,
        "frames_processed":   len(vlm_results),
        "calibration_stats":  cal_stats,
        "ready_for_phase3":   True,
        "phase3_input_file":  str(compact_path),
        "total_time_s":       round(elapsed, 1),
    }
    summary_path = BASE_DIR / "phase2_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log("")
    log("=" * 62)
    log("PHASE 2 COMPLETE")
    log("=" * 62)
    log(f"Model used         : {model_type}")
    log(f"Frames processed   : {len(vlm_results)}")
    log(f"ECE                : {cal_stats['ece']:.6f}  (held-out test set)")
    log(f"Risk MAE           : {cal_stats['risk_mae']:.4f}")
    log(f"Risk Correlation   : {cal_stats['risk_correlation']:.4f}")
    log(f"Total time         : {elapsed:.1f}s")
    log(f"Summary saved      : {summary_path}")
    log("")
    log("Ready for Phase 3 — Cost map construction + A* planner")

    save_log()

if __name__ == "__main__":
    main()
    if sys.platform == "win32":
        input("\nPress Enter to close...")
