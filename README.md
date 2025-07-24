# Fire & People Density Detection – Project README

## Overview

This project offers a modular solution for **real-time fire, smoke, and people detection, tracking, and density analysis** in video streams. It leverages modern YOLOv8 models to provide accurate safety monitoring and crowd analytics for indoor environments.

- **Fire & Smoke Detection**: Utilizes a custom-trained YOLOv8 model (`best.pt`) to identify and annotate fire and smoke occurrences.
- **People Detection**: Uses the standard YOLOv8 model (COCO/pretrained or single-class variant) for detecting and tracking people across video frames.
- **Quadrant-Based Density Metrics**: Each frame is split into quadrants to measure and visualize localized people density and unique person counts.
- **Annotated Visualization**: Draws bounding boxes and overlays statistics for easier interpretation.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Setup & Installation](#setup--installation)
- [Usage Instructions](#usage-instructions)
- [Model Files](#model-files)
- [Expected Output](#expected-output)
- [Customization](#customization)
- [Contact](#contact)

## Features

- **Accurate fire/smoke detection** using a dedicated YOLOv8 model.
- **Real-time people detection and tracking** in video footage.
- **Quadrant-based density reporting** for situational awareness and safety assessment.
- **Clean codebase** leveraging modular utility scripts.
- **Fully annotated video output** for easy validation and demonstration.
- **Seamless handling of custom and COCO model class mappings**.


## Setup & Installation

### 1. Environment

- Python 3.8+
- `pip` (Python package manager)

### 2. Install Dependencies

Navigate to your project directory and install dependencies:

pip install -r requirements.txt


Key dependencies include `ultralytics`, `opencv-python`, and `numpy`.

### 3. Download Models

- Place `best.pt` for fire/smoke detection in the designated models directory.
- Place the YOLOv8 weights (for people detection) as `yolo.pt` or use your preferred YOLOv8 person-only model.

## Usage Instructions

### 1. Running the Script

From your project root, execute:

python main.py


By default, the script:

- Loads `best.pt` for fire/smoke detection.
- Loads `yolo.pt` for people detection (replace as desired).
- Processes `final1.mp4` as the input video (can be changed in `main.py`).

### 2. What the Script Does

- **Frame-by-frame inference** on the specified input video.
- **Draws bounding boxes** with correct labels for fire, smoke, and people.
- **Calculates and displays** normalized density metrics per quadrant, total weighted density, and count of unique tracked people.
- **Visualizes the annotated video** in a resizable window.

### 3. File Configuration

To use a different video, model, or adjust parameters, edit the paths and configuration in `main.py`.

## Model Files

| Model File | Purpose              | Recommended Source/Notes                                                                 |
|------------|---------------------|----------------------------------------------------------------------------------------|
| `best.pt`  | Fire/Smoke detection| YOLOv8 model trained (or downloaded) specifically for "fire" and "smoke" classes.      |
| `yolo.pt`  | People detection    | Standard YOLOv8 COCO (detects "person" and other classes) or a person-only YOLOv8 model.|

## Expected Output

The live video window will show:

- **Red/gray bounding boxes:** Detected fire and smoke
- **Green boxes:** Detected people
- **Density statistics:** Per-quadrant and total, overlaid on frame
- **Unique people count:** Updated in real time

## Customization

- **Switch people detection model:** Use a custom-trained YOLOv8 with only the "person" class for faster, more focused people detection. Adjust `main.py` to use your new model file if needed.
- **Change quadrant logic:** Modify in `detection_utils.py` to customize density analysis granularity.
- **Add new analytics:** Expand utility scripts for area-based statistics or additional incident handling.
- **Configure thresholds:** Tune confidence thresholds in `main.py` for desired sensitivity.

## Contact

For questions, feature suggestions, or collaboration requests, please open an issue on the project repository or contact via GitHub.

---
