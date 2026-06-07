# 🏓 Pickleball Computer Vision Analytics
A computer vision pipeline for analyzing pickleball gameplay videos using deep learning. This tool detects players, ball position, and court layout in each frame to generate rich visual analytics, including heatmaps, rally tempo, and kitchen zone usage. Outputs are compiled into a composite video with multiple synchronized views.

## Table of Contents
[Overview](#overview)

[Features](#features)

[Installation](#installation)

[Usage](#usage)

[Modules](#modules)

## Overview

![](./Demo_Video.gif)

This project combines YOLO-based object/keypoint detection with homography transformations to power analytics to provide a rich breakdown of pickleball gameplay.

Processing Pipeline:

1. Court Detection - Finds 12 keypoints and maps to a bird's-eye-view of the court
2. Player Tracking - Detects players and projects their on-court positions
3. Ball Tracking - Detects the ball each frame and projects it into the birds-eye-space
4. Analytics - Generates player/ball heatmaps, kitchen instrusion panels, and rally tempo tracking
5. Output Rendering - Produces a composite video with synchronized views
## Features
✅ Player Heatmaps – Track player positioning across the match.

✅ Ball Heatmaps – Visualize shot placement and movement.

✅ Kitchen Detection – Shows who is in the non-volley zone per frame.

✅ Rally Length & Tempo – Calculates rally duration and rallies/minute.

✅ Composite Video – Side-by-side view of:

Main annotated video

Bird’s-eye projection

Analytics dashboard (2×2 grid)

✅ Tkinter Desktop App – GUI for video selection, processing progress, and analytics badges.
## Installation
Clone repository
```
git clone URL
cd Pickleball-Analytics
```

Windows setup:
```
uv python install 3.12
uv venv --python 3.12 .venv
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
```

Notes:
- Use Python 3.12 for this project.
- The PyTorch dependencies in `requirements.txt` require the PyTorch CUDA 12.8 wheel index shown above.

## Usage
Run via GUI on Windows
```
.\.venv\Scripts\Activate.ps1
python main.py
```

If PowerShell blocks activation:
```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python main.py
```

1. select a ```.mp4``` video from your system
2. Click process video
3. Progress bar and status will update as processing runs
4. Composite video will be saved under ```video_outputs/run_<timestamp>/Main_overlay.mp4```
## Modules
- [analytics.py](./analytics.py) – Maintains heatmaps, kitchen detection, rally stats; renders analytics panels

- [ball_tracker.py](./ball_tracker.py) – Detects ball bounding box and projects center point into bird’s-eye view

- [court_detection.py](./court_detection.py) – Detects 12 keypoints and computes homography for court projection

- [player_tracker.py](./player_tracker.py) – Detects players and projects their on-court locations

- [process_video.py](./process_video.py) – Orchestrates full pipeline and renders composite video

- [main.py](./main.py) – Tkinter desktop UI for selecting/processing videos and monitoring progress





