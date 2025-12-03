#
# To run:
#
# Install opencv modules in Anaconda environment:
#
#   conda install opencv
#   pip install --upgrade pip
#   pip install opencv-contrib-python
#
# Run calibrate.py from the commandline:
#
#   python calibrate.py

import pickle
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

def calibrate(dir_name):
    calibimgfiles = f'{dir_name}/*.JPG'
    resultfile = 'calibration.pickle'

    # checkerboard coordinates in 3D (7x7 inner corners for 8x8 board)
    objp = np.zeros((7*7,3), np.float32)
    objp[:,:2] = 2.8*np.mgrid[0:7, 0:7].T.reshape(-1,2)

    # arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(calibimgfiles)

    if len(images)==0:
        print('No images found')
        exit()

    # Step through the list and search for chessboard corners
    print(f"Processing {len(images)} calibration images...")
    for idx, fname in enumerate(images):
        print(f"[{idx+1}/{len(images)}] {fname}")
        img = cv2.imread(fname)
        if img is None:
            print(f"  ERROR: Could not read image")
            continue
        img_size = (img.shape[1], img.shape[0])
        
        # Resize for faster corner detection
        scale = 0.25
        img_resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        print(f"  Original: {img.shape[1]}x{img.shape[0]}, Resized: {img_resized.shape[1]}x{img_resized.shape[0]}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners (on resized image)
        # Try with adaptive threshold and normalize
        print(f"  Detecting corners...")
        ret, corners = cv2.findChessboardCorners(gray, (7,7), 
                                                  cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                  cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        # Scale corners back to original image coordinates
        if ret:
            corners = corners / scale
            print(f" Found chessboard corners")
        else:
            print(f" No chessboard corners detected")

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Display image with the corners overlayed
            print('Chessboard corners found')
            cv2.drawChessboardCorners(img, (7,7), corners, ret)
            print('Displaying image with detected corners. Close image window to proceed.')
            cv2.imshow('img', img)
            cv2.waitKey(500)

    # cv2.destroyAllWindows()
    
    print(f"\nSuccessfully detected corners in {len(objpoints)}/{len(images)} images")

    # now perform the calibration
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

    print("Estimated camera intrinsic parameter matrix K")
    print(K)
    print("Estimated radial distortion coefficients")
    print(dist)

    print("Individual intrinsic parameters")
    print("fx = ",K[0][0])
    print("fy = ",K[1][1])
    print("cx = ",K[0][2])
    print("cy = ",K[1][2])


    # save the results out to a file for later use
    calib = {}
    calib["fx"] = K[0][0]
    calib["fy"] = K[1][1]
    calib["cx"] = K[0][2]
    calib["cy"] = K[1][2]
    calib["dist"] = dist
    fid = open(resultfile, "wb" ) 
    pickle.dump(calib,fid)
    fid.close()

    #
    # optionally go through and remove radial distortion from a set of images
    #
    #images = glob.glob(calibimgfiles)
    #for idx, fname in enumerate(images):
    #    img = cv2.imread(fname)
    #    img_size = (img.shape[1], img.shape[0])
    #
    #    dst = cv2.undistort(img, K, dist, None, K)
    #    udfname = fname+'undistort.jpg'
    #    cv2.imwrite(udfname,dst)
    #