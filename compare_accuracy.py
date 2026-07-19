#
# compare_accuracy.py - Side-by-side accuracy comparison of the v3 pipeline
# (camera_positions.json) and the v4 fixed pipeline (camera_positions_v4.json)
# against the tape-measured ground truth from validate.py.

import json
from pathlib import Path

import numpy as np
import yaml

# tape-measured ground truth, single-sourced from ground_truth.yaml
_gt_raw = yaml.safe_load(open(Path(__file__).resolve().parent / 'ground_truth.yaml'))
GT = {label: {side: np.array(_gt_raw[side][label], dtype=float)
              for side in ('left', 'right')}
      for label in ('start', 'end')}


def pos(frame_data, side):
    p = frame_data[f'{side}_camera_position']
    return np.array([p['x'], p['y'], p['z']])


def run_metrics(json_path):
    with open(json_path) as f:
        data = json.load(f)

    m = {'n_frames': len(data)}
    for label, frame_data in (('start', data[0]), ('end', data[-1])):
        for side in ('left', 'right'):
            est = pos(frame_data, side)
            gt = GT[label][side]
            err = np.linalg.norm(est - gt)
            m[f'{label}_{side}_est'] = est
            m[f'{label}_{side}_err'] = err
            m[f'{label}_{side}_pct'] = 100.0 * err / np.linalg.norm(gt)

        est_base = np.linalg.norm(pos(frame_data, 'right') - pos(frame_data, 'left'))
        gt_base = np.linalg.norm(GT[label]['right'] - GT[label]['left'])
        m[f'{label}_baseline_err'] = abs(est_base - gt_base)
        m[f'{label}_baseline_pct'] = 100.0 * abs(est_base - gt_base) / gt_base

    m['mean_abs_err'] = np.mean([m[f'{l}_{s}_err']
                                 for l in ('start', 'end') for s in ('left', 'right')])
    m['mean_pct_err'] = np.mean([m[f'{l}_{s}_pct']
                                 for l in ('start', 'end') for s in ('left', 'right')])

    rms_l = [d['rms_reproj_left_px'] for d in data if 'rms_reproj_left_px' in d]
    rms_r = [d['rms_reproj_right_px'] for d in data if 'rms_reproj_right_px' in d]
    if rms_l:
        m['rms_reproj_mean'] = np.mean(rms_l + rms_r)
        m['rms_reproj_max'] = np.max(rms_l + rms_r)
    return m


def fmt_vec(v):
    return f"({v[0]:7.1f}, {v[1]:7.1f}, {v[2]:7.1f})"


def main():
    base = Path(__file__).resolve().parent / 'data' / 'checkerboard2' / 'videos'
    v3 = run_metrics(base / 'camera_positions.json')
    v4 = run_metrics(base / 'camera_positions_v4.json')

    print(f"{'':38s}{'v3 (old)':>14s}{'v4 (fixed)':>14s}{'delta':>12s}")
    print('-' * 78)

    rows = [
        ('Start: left cam abs error (cm)', 'start_left_err'),
        ('Start: right cam abs error (cm)', 'start_right_err'),
        ('End:   left cam abs error (cm)', 'end_left_err'),
        ('End:   right cam abs error (cm)', 'end_right_err'),
        ('Start: baseline error (cm)', 'start_baseline_err'),
        ('End:   baseline error (cm)', 'end_baseline_err'),
        ('Mean abs positioning error (cm)', 'mean_abs_err'),
        ('Mean positioning error (%)', 'mean_pct_err'),
    ]
    for label, key in rows:
        d = v4[key] - v3[key]
        print(f"{label:38s}{v3[key]:14.2f}{v4[key]:14.2f}{d:+12.2f}")

    print('-' * 78)
    print("\nEstimated positions vs ground truth (cm):")
    for label in ('start', 'end'):
        for side in ('left', 'right'):
            gt = GT[label][side]
            print(f"  {label:5s} {side:5s}  GT {fmt_vec(gt)}   "
                  f"v3 {fmt_vec(v3[f'{label}_{side}_est'])}   "
                  f"v4 {fmt_vec(v4[f'{label}_{side}_est'])}")

    if 'rms_reproj_mean' in v4:
        print(f"\nv4 RMS reprojection error: mean {v4['rms_reproj_mean']:.2f} px, "
              f"max {v4['rms_reproj_max']:.2f} px "
              f"(v3 did not record reprojection error)")
    print(f"Frames: v3={v3['n_frames']}, v4={v4['n_frames']}")

    # scale diagnostic: a consistent |est|/|gt| ratio across endpoints is the
    # signature of a wrong physical square size (or GT scale), not pose error
    keys = [(l, s) for l in ('start', 'end') for s in ('left', 'right')]
    ratios = [np.linalg.norm(v4[f'{l}_{s}_est']) / np.linalg.norm(GT[l][s])
              for l, s in keys]
    scales = np.linspace(0.7, 1.2, 2001)
    errs = [np.mean([np.linalg.norm(sc*v4[f'{l}_{s}_est'] - GT[l][s])
                     for l, s in keys]) for sc in scales]
    best = scales[int(np.argmin(errs))]
    sq = yaml.safe_load(open(Path(__file__).resolve().parent
                             / 'ground_truth.yaml'))['square_size_cm']
    print(f"\nScale diagnostic (v4): |est|/|gt| ratios = "
          + ", ".join(f"{r:.3f}" for r in ratios))
    print(f"  best single scale {best:.3f} -> mean error {min(errs):.1f} cm "
          f"(implied square size {sq*best:.3f} cm instead of {sq} cm)")


if __name__ == '__main__':
    main()
