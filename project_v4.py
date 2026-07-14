#
# project_v4.py - Camera localization pipeline with the fixes from
# local-docs/03-bugs.md applied, plus a global branch-disambiguation step
# that turned out to be necessary on this data. Headless: writes
# camera_positions_v4.json instead of showing interactive plots.
#
# Fixes vs project_v3.py:
#   1. Lens distortion applied: corners undistorted with cv2.undistortPoints
#   2. cornerSubPix refinement with a window scaled to the corner spacing
#      (the board is only ~60-90 px wide in these videos; a large window
#      pulls in neighboring corners and corrupts the detection)
#   3. Checkerboard 180-degree ambiguity resolved globally. An 8x6 board is
#      exactly symmetric under a 180-degree rotation (corner grid AND square
#      coloring), so every frame has mirror-pose solutions with identical
#      reprojection error, and findChessboardCorners silently reverses its
#      corner ordering as the view direction changes. Near the board's
#      symmetry axis a warm-started sequential tracker can drift onto the
#      mirror branch (this is what corrupted the right camera's second half
#      in v3). Instead of tracking sequentially, we enumerate all candidate
#      poses per frame with cv2.solvePnPGeneric (IPPE gives both planar-pose
#      solutions, for both corner orderings) and pick the globally smoothest
#      track through the sequence by dynamic programming, anchored at the
#      tape-measured start pose. No hand-tuned initial guess and no
#      warm-start/smoothness machinery is needed after this.
#   4. findChessboardCorners return values checked; skipped frames logged;
#      real frame indices/timestamps preserved in the output JSON
#   5. Per-frame RMS reprojection error reported and saved
#   6. leastsq convergence flag (ier) checked
#
# Anchor poses below are the tape-measured start positions from validate.py
# (v3's hand-typed init poses had the z sign flipped and x values swapped).

from pathlib import Path
import copy
import json
import os
import pickle

import cv2
import numpy as np
import scipy.optimize
import yaml
from scipy.spatial.transform import Rotation as SciRot

from camutils import Camera, makerotation, triangulate
from calibrate_v3 import calibrate

_GT = yaml.safe_load(open(Path(__file__).resolve().parent / 'ground_truth.yaml'))

BOARD_SIZE = tuple(_GT['board_inner_corners'])   # inner corners (cols, rows)
SQUARE_CM = float(_GT['square_size_cm'])
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# [rx, ry, rz, tx, ty, tz] - tape-measured start poses, used only to anchor
# the branch disambiguation at frame 0
ANCHOR_POSE_L = np.array(_GT['anchor_rotation_deg'] + _GT['left']['start'], dtype=float)
ANCHOR_POSE_R = np.array(_GT['anchor_rotation_deg'] + _GT['right']['start'], dtype=float)


def rotation_angle_deg(Ra, Rb):
    """Geodesic angle between two rotation matrices, in degrees."""
    cos = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def residuals(pts3, pts2, cam, params):
    cam.update_extrinsics(params)
    return (pts2 - cam.project(pts3)).flatten()


def params_from_pose(Rw, tw):
    """Convert a world-frame pose (R, t) to the [rx,ry,rz,tx,ty,tz] vector
    expected by camutils.makerotation. Note makerotation's Ry uses the
    opposite sign convention from scipy's extrinsic 'xyz', so ry is negated.
    """
    a, b, c = SciRot.from_matrix(Rw).as_euler('xyz', degrees=True)
    return np.array([a, -b, c, tw[0, 0], tw[1, 0], tw[2, 0]])


def board_points():
    """3D coordinates of the checkerboard inner corners (world frame, z=0)."""
    pts3 = np.zeros((3, BOARD_SIZE[0]*BOARD_SIZE[1]))
    xx, yy = np.meshgrid(np.arange(BOARD_SIZE[0]), np.arange(BOARD_SIZE[1]))
    pts3[0, :] = SQUARE_CM*xx.reshape(1, -1)
    pts3[1, :] = SQUARE_CM*yy.reshape(1, -1)
    return pts3


def detect_corners(img_path, K, dist, K_pix):
    """Detect chessboard corners, refine to subpixel at full resolution and
    undistort them into the square-pixel camera K_pix (fx = fy = f_avg), so
    the single-focal camutils Camera model is exact rather than approximate.
    Returns a (2,N) array or None if detection failed."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)
    if not ret:
        return None
    # the refinement window must stay well under the corner spacing, or it
    # pulls in neighboring corners (the board is only ~60-90 px wide here)
    spacing = np.median(np.linalg.norm(np.diff(corners.squeeze(), axis=0), axis=1))
    half = int(np.clip(spacing * 0.4, 2, 11))
    corners = cv2.cornerSubPix(gray, corners, (half, half), (-1, -1), SUBPIX_CRITERIA)
    corners = cv2.undistortPoints(corners, K, dist, P=K_pix)
    return corners.squeeze().T


def pnp_candidates(pts3, pts2, K):
    """All plausible poses for one frame: solvePnPGeneric/IPPE returns both
    planar-pose solutions, computed for both corner orderings (2x2 = up to 4
    candidates). Poses are converted to world frame (R = camera-to-world
    rotation, t = camera center), matching the camutils Camera convention.
    """
    obj = np.ascontiguousarray(pts3.T.reshape(-1, 1, 3), dtype=np.float64)
    cands = []
    for flipped, p2 in ((False, pts2), (True, pts2[:, ::-1])):
        img = np.ascontiguousarray(p2.T.reshape(-1, 1, 2), dtype=np.float64)
        try:
            n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
                obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
        except cv2.error:
            continue
        errs = np.ravel(errs) if errs is not None else [np.nan]*len(rvecs)
        for rvec, tvec, err in zip(rvecs, tvecs, errs):
            Rcv, _ = cv2.Rodrigues(rvec)
            cands.append({
                'R': Rcv.T,
                't': -Rcv.T @ tvec.reshape(3, 1),
                'err': float(err),
                'flipped': flipped,
                'pts2': p2,
            })
    return cands


def choose_track(cand_lists, anchor_params):
    """Pick one pose candidate per frame so the whole track is as smooth as
    possible (dynamic programming / Viterbi over candidate poses, transition
    cost = rotation angle in degrees + translation distance in cm), anchored
    to the measured start pose. Resolves the board-symmetry branch ambiguity
    that per-frame or frame-to-frame methods cannot."""
    def pose_dist(Ra, ta, Rb, tb):
        return rotation_angle_deg(Ra, Rb) + float(np.linalg.norm(ta - tb))

    R0 = makerotation(*anchor_params[0:3])
    t0 = anchor_params[3:6].reshape(3, 1)

    costs = [[pose_dist(R0, t0, c['R'], c['t']) for c in cand_lists[0]]]
    back = [[0]*len(cand_lists[0])]
    for k in range(1, len(cand_lists)):
        ck, bk = [], []
        for c in cand_lists[k]:
            trans = [costs[k-1][j] + pose_dist(p['R'], p['t'], c['R'], c['t'])
                     for j, p in enumerate(cand_lists[k-1])]
            j_best = int(np.argmin(trans))
            ck.append(trans[j_best])
            bk.append(j_best)
        costs.append(ck)
        back.append(bk)

    path = [int(np.argmin(costs[-1]))]
    for k in range(len(cand_lists)-1, 0, -1):
        path.append(back[k][path[-1]])
    return path[::-1]


def refine_pose(pts3, cand, cam):
    """Levenberg-Marquardt refinement of a candidate pose (pure reprojection
    error, camutils camera model). Returns (cam, rms_px)."""
    p0 = params_from_pose(cand['R'], cand['t'])
    popt, _, _, mesg, ier = scipy.optimize.leastsq(
        lambda p: residuals(pts3, cand['pts2'], cam, p), p0, full_output=True)
    if ier not in (1, 2, 3, 4):
        print(f"  WARNING: leastsq did not converge (ier={ier}): {mesg}")
    cam.update_extrinsics(popt)
    rms = float(np.sqrt(np.mean(residuals(pts3, cand['pts2'], cam, popt)**2)))
    return cam, rms


def calibrate_intr(dir_name, video_path):
    """Intrinsic calibration from video frames (reuses frames/pickle if present).

    Returns (camL, camR, K, dist).
    """
    if not os.path.exists(dir_name) or not os.listdir(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = int(frame_count / fps)
        print(f"Video FPS: {fps}, Total Frames: {frame_count}, Duration (s): {duration}")
        for i in range(duration):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * fps))
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(os.path.join(dir_name, f'frame_{i}.jpg'), frame)
        cap.release()

    pickle_path = Path(__file__).resolve().parent / 'calibration_v4.pickle'
    if not pickle_path.exists():
        calibrate(dir_name, pickle_file=str(pickle_path))
    with open(pickle_path, 'rb') as f:
        calib = pickle.load(f)

    print(f"Intrinsics: fx={calib['fx']:.1f} fy={calib['fy']:.1f} "
          f"cx={calib['cx']:.1f} cy={calib['cy']:.1f} "
          f"RMS={calib.get('rms', float('nan')):.3f} px")

    f_avg = (calib['fx'] + calib['fy'])/2
    c = np.array([[calib['cx'], calib['cy']]]).T
    K = calib.get('K')
    if K is None:
        K = np.array([[calib['fx'], 0, calib['cx']],
                      [0, calib['fy'], calib['cy']],
                      [0, 0, 1]])
    dist = np.asarray(calib['dist'])

    # square-pixel working camera: corners are undistorted into this matrix,
    # making the single-focal camutils Camera model exact
    K_pix = np.array([[f_avg, 0, calib['cx']],
                      [0, f_avg, calib['cy']],
                      [0, 0, 1]])

    R_init = np.eye(3)
    t_init = np.zeros((3, 1))
    camL = Camera(f_avg, c, R_init, t_init)
    camR = Camera(f_avg, c, R_init, t_init)
    return camL, camR, K, dist, K_pix


def extract_frame_pairs(dir_name, intervals=0.5):
    """Sample synchronized frame pairs from left.MP4/right.MP4 every
    `intervals` seconds. Reuses videos/extracted if already populated."""
    extracted_dir = os.path.join(dir_name, 'extracted')

    capL = cv2.VideoCapture(os.path.join(dir_name, 'left.MP4'))
    capR = cv2.VideoCapture(os.path.join(dir_name, 'right.MP4'))
    fpsL = capL.get(cv2.CAP_PROP_FPS)
    fpsR = capR.get(cv2.CAP_PROP_FPS)
    durationL = int(capL.get(cv2.CAP_PROP_FRAME_COUNT)) / fpsL
    durationR = int(capR.get(cv2.CAP_PROP_FRAME_COUNT)) / fpsR
    num_frames = int(min(durationL, durationR) / intervals)
    print(f"Left: {fpsL:.2f} fps {durationL:.2f}s | Right: {fpsR:.2f} fps "
          f"{durationR:.2f}s | {num_frames} frame pairs at {intervals}s")

    frame_pairs = []
    reuse = os.path.isdir(extracted_dir) and all(
        os.path.exists(os.path.join(extracted_dir, f'{side}_frame_{i}.jpg'))
        for i in range(num_frames) for side in ('left', 'right'))

    if reuse:
        print(f"Reusing extracted frames in {extracted_dir}")
        for i in range(num_frames):
            frame_pairs.append((i,
                                os.path.join(extracted_dir, f'left_frame_{i}.jpg'),
                                os.path.join(extracted_dir, f'right_frame_{i}.jpg')))
    else:
        os.makedirs(extracted_dir, exist_ok=True)
        for i in range(num_frames):
            time_stamp = i * intervals
            capL.set(cv2.CAP_PROP_POS_FRAMES, int(time_stamp * fpsL))
            retL, frameL = capL.read()
            capR.set(cv2.CAP_PROP_POS_FRAMES, int(time_stamp * fpsR))
            retR, frameR = capR.read()
            if retL and retR:
                left_path = os.path.join(extracted_dir, f'left_frame_{i}.jpg')
                right_path = os.path.join(extracted_dir, f'right_frame_{i}.jpg')
                cv2.imwrite(left_path, frameL)
                cv2.imwrite(right_path, frameR)
                frame_pairs.append((i, left_path, right_path))
            else:
                print(f"  Could not read frame pair {i} at {time_stamp:.1f}s")

    capL.release()
    capR.release()
    return frame_pairs, intervals


def localize(dir_name, camL, camR, K, dist, K_pix, intervals=0.5):
    """Per-frame candidate poses -> global branch disambiguation -> refine."""
    frame_pairs, intervals = extract_frame_pairs(dir_name, intervals)
    pts3_board = board_points()

    # pass 1: detect corners and enumerate pose candidates per frame
    frames = []
    skipped = []
    for frame_idx, left_path, right_path in frame_pairs:
        pts2L = detect_corners(left_path, K, dist, K_pix)
        pts2R = detect_corners(right_path, K, dist, K_pix)
        candsL = pnp_candidates(pts3_board, pts2L, K_pix) if pts2L is not None else []
        candsR = pnp_candidates(pts3_board, pts2R, K_pix) if pts2R is not None else []
        if not candsL or not candsR:
            side = 'left' if not candsL else 'right'
            print(f"Frame {frame_idx}: detection/PnP failed ({side}), skipping")
            skipped.append(frame_idx)
            continue
        frames.append({'idx': frame_idx, 'candsL': candsL, 'candsR': candsR})

    if not frames:
        print("No frames processed")
        return []

    # pass 2: globally smoothest track through the pose candidates
    pathL = choose_track([f['candsL'] for f in frames], ANCHOR_POSE_L)
    pathR = choose_track([f['candsR'] for f in frames], ANCHOR_POSE_R)

    # pass 3: LM refinement of the chosen poses + stereo triangulation check
    results = []
    for f, jL, jR in zip(frames, pathL, pathR):
        candL, candR = f['candsL'][jL], f['candsR'][jR]
        camL_i, rmsL = refine_pose(pts3_board, candL, copy.deepcopy(camL))
        camR_i, rmsR = refine_pose(pts3_board, candR, copy.deepcopy(camR))

        pts3_tri = triangulate(candL['pts2'], camL_i, candR['pts2'], camR_i)
        tri_err = float(np.sqrt(np.mean((pts3_tri - pts3_board)**2)))

        frame_idx = f['idx']
        print(f"Frame {frame_idx} (t={frame_idx*intervals:.1f}s): "
              f"RMS reproj L={rmsL:.2f}px R={rmsR:.2f}px | "
              f"board triangulation RMS={tri_err:.2f}cm | "
              f"ordering L={'rev' if candL['flipped'] else 'raw'} "
              f"R={'rev' if candR['flipped'] else 'raw'}")

        results.append({
            'frame': frame_idx,
            'timestamp_seconds': frame_idx * intervals,
            'left_camera_position': {
                'x': float(camL_i.t[0, 0]),
                'y': float(camL_i.t[1, 0]),
                'z': float(camL_i.t[2, 0]),
            },
            'right_camera_position': {
                'x': float(camR_i.t[0, 0]),
                'y': float(camR_i.t[1, 0]),
                'z': float(camR_i.t[2, 0]),
            },
            'left_camera_rotation': camL_i.R.tolist(),
            'right_camera_rotation': camR_i.R.tolist(),
            'rms_reproj_left_px': rmsL,
            'rms_reproj_right_px': rmsR,
            'board_triangulation_rms_cm': tri_err,
        })

    if skipped:
        print(f"\nSkipped {len(skipped)} frame pairs: {skipped}")
    else:
        print(f"\nAll {len(results)} frame pairs processed, none skipped")

    json_path = os.path.join(dir_name, 'camera_positions_v4.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {json_path}")
    return results


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent
    calib_dir = base_dir / 'data' / 'checkerboard2' / 'calibrate'

    camL, camR, K, dist, K_pix = calibrate_intr(
        str(calib_dir / 'pics_from_vid'),
        str(calib_dir / 'video' / 'calibrate.MP4'))

    videos_dir = base_dir / 'data' / 'checkerboard2' / 'videos'
    results = localize(str(videos_dir), camL, camR, K, dist, K_pix)
    print(f"\nProcessed {len(results)} frame pairs successfully")
