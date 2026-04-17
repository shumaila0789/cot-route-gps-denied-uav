# CoT-Route: Uncertainty-Aware Semantic Routing for GPS-Denied UAV Navigation

## Requirements
Python 3.12, CPU only, ~64GB RAM recommended

## Installation
pip install -r requirements.txt

## Dataset
Download TartanAir from [official link] and place under tartanair/

## Quick Start (using pre-computed data)
python phase3_planner.py
# Reads pre-built graph and VLM outputs — no GPU or VLM needed

## Full Pipeline Reproduction
python phase1_data_pipeline.py
python phase2_vlm_reasoning.py   # Warning: slow on CPU (~hours)
python find_bottleneck_scenarios.py
python phase3_planner.py

## Results
See results/ for CSV tables matching Tables II and III in the paper.
