#
# calibrate_v3.py - Fixed intrinsic calibration (see local-docs/03-bugs.md)
#
# Changes vs calibrate_v2.py:
#   - cornerSubPix refinement runs on the FULL-resolution grayscale image
#     (v2 refined on the 0.5x image and scaled up, halving the precision)
#   - RMS reprojection error from cv2.calibrateCamera is printed and saved
#   - Full K matrix and distortion coefficients saved to the pickle
#   - No GUI windows unless show=True; raises instead of exit()

import pickle
import numpy as np
import cv2
import glob


def calibrate(dir_name, pickle_file, show=False):
    calibimgfiles = f'{dir_name}/*.jpg'

    # checkerboard coordinates in 3D (2.8 cm squares, board plane z=0)
    objp = np.zeros((8*6, 3), np.float32)
    objp[:, :2] = 2.8*np.mgrid[0:8, 0:6].T.reshape(-1, 2)

    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane

    images = glob.glob(calibimgfiles)
    if len(images) == 0:
        raise FileNotFoundError(f'No calibration images found in {calibimgfiles}')

    img_size = None
    print(f"Processing {len(images)} calibration images...")
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            print(f"[{idx+1}/{len(images)}] {fname}: ERROR could not read image")
            continue
        img_size = (img.shape[1], img.shape[0])

        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detect on a 0.5x image for speed
        scale = 0.5
        gray_small = cv2.resize(gray_full, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_AREA)
        ret, corners = cv2.findChessboardCorners(gray_small, (8, 6),
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                 cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                 cv2.CALIB_CB_FAST_CHECK)
        if not ret:
            print(f"[{idx+1}/{len(images)}] {fname}: no chessboard corners detected")
            continue

        # refine at full resolution
        corners = (corners / scale).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray_full, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners)
        print(f"[{idx+1}/{len(images)}] {fname}: corners found and refined")

        if show:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, (8, 6), corners, ret)
            display_img = cv2.resize(vis, None, fx=0.2, fy=0.2,
                                     interpolation=cv2.INTER_AREA)
            cv2.imshow('img', display_img)
            cv2.waitKey(500)

    if show:
        cv2.destroyAllWindows()

    if not objpoints:
        raise RuntimeError('Chessboard corners were not detected in any image')

    print(f"\nSuccessfully detected corners in {len(objpoints)}/{len(images)} images")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                     img_size, None, None)

    print(f"RMS reprojection error: {rms:.4f} px")
    print("Estimated camera intrinsic parameter matrix K")
    print(K)
    print("Estimated distortion coefficients [k1 k2 p1 p2 k3]")
    print(dist)

    calib = {
        'fx': K[0][0], 'fy': K[1][1],
        'cx': K[0][2], 'cy': K[1][2],
        'K': K, 'dist': dist,
        'rms': rms, 'img_size': img_size,
    }
    with open(pickle_file, 'wb') as fid:
        pickle.dump(calib, fid)

    return calib
