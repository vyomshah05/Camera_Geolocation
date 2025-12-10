# Camera Geolocation Project

## How to Run

```bash
pip install -r requirements.txt

python project_v2.py

python project_v3.py
```

## Files

**project_v3.py** - Script to perform camera localization from video (calibrates intrinsic and extrinsic from video). Shows plot with slider on the bottom to move through frames. Prints error stats to terminal.


**project_v2.py** - Script to perform camera localization from images. Shows plot with multiple view angles and prints error stats in terminal.

**visualization.gif** - gif of the plot generated from project_v3.py

**camera_animation.gif** - gif of selection of frames from Left and Right Camera for project_v3.py

**validate.py** - Runs error analysis

**calibrate_v2.py** - Modified calibrate script provided from Assignment 3

**camutils.py** and **visutils.py** - Provided from Assignment 3

**attributes.yaml** - Picture and video location and data

## Data

**data/checkerboard2/** - Main 6x8 checkerboard used for project_v2 and project_v3

**data/checkerboard2/videos/** - Contains:
- `left.MP4` - Left camera video for project_v3
- `right.MP4` - Right camera video for project_v3
- `camera_positions.json` - JSON storing all of the location data for project_v3

**data/checkerboard2/positions1/** - Images for extrinsic calibration/localization for project_v2

**data/checkerboard2/calibrate/2/** - Images for intrinsic calibration for project_v2

**data/checkerboard2/calibrate/video/** - Video for intrinsic calibration for project_v3

**data/checkerboard1/** - Failed 7x7 checkerboard
