#
# make_camera_gif.py - Build a side-by-side gif of the left.MP4 / right.MP4
# frames actually used by the v4 pipeline (data/checkerboard2/videos/extracted,
# sampled every 0.5 s), on the same timeline and frame rate as
# visualization_v4.gif so the two can be compared frame by frame.
#
#   python make_camera_gif.py    # writes camera_v4.gif
#

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PANEL_WIDTH = 400   # px per camera panel
FPS = 4             # matches visualization_v4.gif
INTERVAL_S = 0.5    # sampling interval used by project_v4.py


def annotate(img, label):
    cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    base_dir = Path(__file__).resolve().parent
    extracted = base_dir / 'data' / 'checkerboard2' / 'videos' / 'extracted'

    frames = []
    i = 0
    while True:
        left_path = extracted / f'left_frame_{i}.jpg'
        right_path = extracted / f'right_frame_{i}.jpg'
        if not (left_path.exists() and right_path.exists()):
            break
        panels = []
        for path, label in ((left_path, 'left.MP4 (blue)'),
                            (right_path, 'right.MP4 (red)')):
            img = cv2.imread(str(path))
            scale = PANEL_WIDTH / img.shape[1]
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
            panels.append(annotate(img, label))
        combo = cv2.hconcat(panels)
        annotate(combo, '')
        cv2.putText(combo, f't={i*INTERVAL_S:.1f}s',
                    (combo.shape[1]//2 - 40, combo.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(combo, f't={i*INTERVAL_S:.1f}s',
                    (combo.shape[1]//2 - 40, combo.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        frames.append(Image.fromarray(cv2.cvtColor(combo, cv2.COLOR_BGR2RGB)))
        i += 1

    if not frames:
        raise SystemExit(f'No extracted frames found in {extracted} - run project_v4.py first')

    out = base_dir / 'camera_v4.gif'
    frames[0].save(str(out), save_all=True, append_images=frames[1:],
                   duration=int(1000/FPS), loop=0, optimize=True)
    print(f"Wrote {out} ({i} frames, {out.stat().st_size/1e6:.1f} MB)")


if __name__ == '__main__':
    main()
