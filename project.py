from camutils import Camera, residuals, calibratePose, triangulate
from calibrate import calibrate
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import visutils
import cv2
import pickle


def calibrate_intr(dir_name):
    """ Calibrate the camera using images in the specified directory.
    
    Args:
        dir_name (str): Directory containing calibration images.
    Returns:
       Camera: Calibrated camera object.
    """
    # perform calibration to get intrinsic parameters
    #calibrate(dir_name)
    # load in the calibration parameters
    with open('calibration.pickle','rb') as f:
        calib_p = pickle.load(f)
    # extract intrinsic parameters
    f = (calib_p['fx']+calib_p['fy'])/2
    c = np.array([[calib_p['cx'],calib_p['cy']]]).T
    # create Camera objects representing the left and right cameras
    # use the known intrinsic parameters you loaded in.
    R_init = np.array([[1,0,0],[0,1,0],[0,0,1]])
    t_init = np.array([[0,0,0]]).T
    cam = Camera(f,c,R_init,t_init)

    return cam

def calibrate_extr(camL, camR, file_pathL, file_pathR):
    """ Calibrate the extrinsic parameters of the camera given 3D-2D point correspondences.
    
    Args:
        cam (Camera): Camera object with known intrinsic parameters.
        file_path (str): Path to the file containing 3D-2D point correspondences.
    Returns:
        (CameraL, CameraR): Camera objects with updated extrinsic parameters.
    """
    # optimize the extrinsic parameters to minimize reprojection error
    imgL = plt.imread(file_pathL)
    retL, cornersL = cv2.findChessboardCorners(imgL, (7,7), None)
    imgL = plt.imread('file_pathL')
    retL, cornersL = cv2.findChessboardCorners(imgL, (7,7), None)
    pts2L = cornersL.squeeze().T
    imgR = plt.imread(file_pathR)
    retR, cornersR = cv2.findChessboardCorners(imgR, (7,7), None)
    imgR = plt.imread('file_pathR')
    retR, cornersR = cv2.findChessboardCorners(imgR, (7,7), None)
    pts2R = cornersR.squeeze().T
    
    pts3 = np.zeros((3,7*7))
    yy,xx = np.meshgrid(np.arange(7),np.arange(7))
    pts3[0,:] = 2.8*xx.reshape(1,-1)
    pts3[1,:] = 2.8*yy.reshape(1,-1)

    camL = calibratePose(pts3,pts2L,camL,np.array([0,0,0,0,0,-2]))
    camR = calibratePose(pts3,pts2R,camR,np.array([0,0,0,0,0,-2]))

    return camL, camR

if __name__ == '__main__':
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent
    dir_path = base_dir / 'data' / 'checkerboard1'
    cam = calibrate_intr(str(dir_path))
    print(cam)
    print(camL)
    print(camR)
    
    #Plot the camera positions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    visutils.label_axes(ax)
    visutils.set_axes_equal_3d(ax)
    ax.scatter(camL.t[0],camL.t[1],camL.t[2],c='r',marker='o')
    ax.scatter(camR.t[0],camR.t[1],camR.t[2],c='b',marker='o')
    plt.show()