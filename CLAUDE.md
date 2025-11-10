# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyBullet-based race track simulator with camera line extraction for autonomous vehicle testing. The camera captures track boundaries using PyBullet's segmentation masks (not color filtering), outputting clean binary images of track lines for computer vision algorithms.

## Running the Simulation

```bash
# Install dependencies
uv pip install -r requirements.txt
# or with pip
pip install -r requirements.txt

# Run the simulation
cd src
python main.py
```

**Controls**:
- Arrow Keys: Tank drive control
  - `↑`: Forward
  - `↓`: Backward
  - `←`: Rotate left
  - `→`: Rotate right
  - Combinations: Forward/Backward + Left/Right for turning while moving
- `Q`: Quit

## Architecture

### Critical Design Pattern: Segmentation Mask-Based Line Extraction

The track line extraction relies on PyBullet's segmentation masks, NOT color filtering. This is a key architectural constraint