#
# visualize_v4.py - Interactive visualization of the v4 localization results.
# Reads camera_positions_v4.json (positions + rotations written by
# project_v4.py), no reprocessing needed.
#
#   python visualize_v4.py          # interactive figure with frame slider
#   python visualize_v4.py --gif    # render visualization_v4.gif headless
#

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import yaml


def load_results(base_dir):
    json_path = base_dir / 'data' / 'checkerboard2' / 'videos' / 'camera_positions_v4.json'
    with open(json_path) as f:
        results = json.load(f)
    if 'left_camera_rotation' not in results[0]:
        raise SystemExit('camera_positions_v4.json has no rotations - rerun project_v4.py')
    return results


def board_points(base_dir):
    gt = yaml.safe_load(open(base_dir / 'ground_truth.yaml'))
    ncols, nrows = gt['board_inner_corners']
    sq = gt['square_size_cm']
    pts3 = np.zeros((3, ncols*nrows))
    xx, yy = np.meshgrid(np.arange(ncols), np.arange(nrows))
    pts3[0, :] = sq*xx.reshape(1, -1)
    pts3[1, :] = sq*yy.reshape(1, -1)
    return pts3


def frame_geometry(frame):
    out = {}
    for side in ('left', 'right'):
        p = frame[f'{side}_camera_position']
        t = np.array([[p['x']], [p['y']], [p['z']]])
        R = np.array(frame[f'{side}_camera_rotation'])
        look = np.hstack((t, t + R @ np.array([[0, 0, 50]]).T))
        out[side] = (t, R, look)
    return out


def make_figure(results, pts3, interactive=True):
    import matplotlib.pyplot as plt
    import visutils

    fig = plt.figure(figsize=(16, 10))
    plt.subplots_adjust(bottom=0.12, top=0.92, left=0.05, right=0.95,
                        hspace=0.3, wspace=0.3)
    ax_3d = fig.add_subplot(2, 2, 1, projection='3d')
    ax_xz = fig.add_subplot(2, 2, 2)
    ax_yz = fig.add_subplot(2, 2, 3)
    ax_xy = fig.add_subplot(2, 2, 4)

    def update_plot(frame_idx):
        frame_idx = int(frame_idx)
        frame = results[frame_idx]
        g = frame_geometry(frame)
        tL, _, lookL = g['left']
        tR, _, lookR = g['right']
        pL, pR = tL.flatten(), tR.flatten()
        dist = np.linalg.norm(pL - pR)
        mid = (pL + pR) / 2.0
        d = pR - pL

        for ax in (ax_3d, ax_xz, ax_yz, ax_xy):
            ax.clear()

        fig.suptitle(f"Frame {frame['frame']} - v4 Camera Localization "
                     f"(t={frame['timestamp_seconds']:.1f}s)",
                     fontsize=14, fontweight='bold')

        # 3D view
        ax_3d.view_init(elev=-58, azim=51, roll=43)
        ax_3d.scatter(pts3[0, :], pts3[1, :], pts3[2, :], c='k', marker='x',
                      label='Checkerboard')
        ax_3d.plot(tR[0], tR[1], tR[2], 'ro', markersize=10, label='Right Camera')
        ax_3d.plot(tL[0], tL[1], tL[2], 'bo', markersize=10, label='Left Camera')
        ax_3d.plot(lookL[0, :], lookL[1, :], lookL[2, :], 'b-', linewidth=2)
        ax_3d.plot(lookR[0, :], lookR[1, :], lookR[2, :], 'r-', linewidth=2)
        ax_3d.plot([pL[0], pR[0]], [pL[1], pR[1]], [pL[2], pR[2]], '--',
                   color='gray', linewidth=1)
        ax_3d.text(mid[0], mid[1], mid[2],
                   f"x diff: {d[0]:.1f} cm\ny diff: {d[1]:.1f} cm\n"
                   f"z diff: {d[2]:.1f} cm\ndistance: {dist:.1f} cm",
                   color='purple',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
        visutils.set_axes_equal_3d(ax_3d)
        visutils.label_axes(ax_3d)
        ax_3d.set_title('Scene 3D View')
        ax_3d.legend()

        # orthographic views: (title, horizontal fn, vertical fn, labels)
        views = [
            (ax_xz, 'XZ-view (Top View)', lambda v: v[0], lambda v: -v[2],
             'x (cm)', 'z (cm)', f"x diff: {d[0]:.1f} cm\nz diff: {d[2]:.1f} cm"),
            (ax_yz, 'YZ-view (Side View)', lambda v: -v[2], lambda v: v[1],
             '-z (cm)', 'y (cm)', f"y diff: {d[1]:.1f} cm\nz diff: {d[2]:.1f} cm"),
            (ax_xy, 'XY-view (Front View from Checkerboard)',
             lambda v: v[0], lambda v: v[1],
             'x (cm)', 'y (cm)', f"x diff: {d[0]:.1f} cm\ny diff: {d[1]:.1f} cm"),
        ]
        for ax, title, fh, fv, xl, yl, txt in views:
            ax.plot(fh(pts3), fv(pts3), 'k.', label='Checkerboard')
            ax.plot(fh(pR), fv(pR), 'ro', markersize=10, label='Right Camera')
            ax.plot(fh(pL), fv(pL), 'bo', markersize=10, label='Left Camera')
            ax.plot(fh(lookL), fv(lookL), 'b-', linewidth=2)
            ax.plot(fh(lookR), fv(lookR), 'r-', linewidth=2)
            ax.plot([fh(pL), fh(pR)], [fv(pL), fv(pR)], '--',
                    color='gray', linewidth=1)
            ax.text(fh(mid), fv(mid), txt, color='purple',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
            ax.set_title(title)
            ax.grid()
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.legend()
            ax.axis('equal')

        if interactive:
            fig.canvas.draw_idle()

    return fig, update_plot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gif', action='store_true',
                        help='render visualization_v4.gif instead of showing UI')
    args = parser.parse_args()

    if args.gif:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    base_dir = Path(__file__).resolve().parent
    results = load_results(base_dir)
    pts3 = board_points(base_dir)
    fig, update_plot = make_figure(results, pts3, interactive=not args.gif)

    if args.gif:
        from matplotlib.animation import FuncAnimation, PillowWriter
        anim = FuncAnimation(fig, update_plot, frames=len(results))
        out = base_dir / 'visualization_v4.gif'
        anim.save(str(out), writer=PillowWriter(fps=4), dpi=60)
        print(f"Wrote {out}")
        return

    update_plot(0)
    ax_slider = plt.axes([0.15, 0.02, 0.7, 0.03])
    slider = Slider(ax=ax_slider, label='Frame', valmin=0,
                    valmax=len(results) - 1, valinit=0, valstep=1)
    slider.on_changed(update_plot)
    plt.show()


if __name__ == '__main__':
    main()
