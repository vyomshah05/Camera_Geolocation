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
import cv2
import os


def calibrate_intr(dir_name):
    """ Calibrate the camera using images in the specified directory.
    
    Args:
        dir_name (str): Directory containing calibration images.
    Returns:
       (CameraL, CameraR): Calibrated camera object.
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
    camL = Camera(f,c,R_init,t_init)
    camR = Camera(f,c,R_init,t_init)

    return (camL, camR)

def calibrate_from_vid(dir_name, video_path):
    """ Calibrate the camera using images extracted from a video in the specified directory.
    
    Args:
        dir_name (str): Directory containing calibration video.
    Returns:
       (CameraL, CameraR): Calibrated camera object.
    """ 

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(frame_count / fps)
    
    #extract frames at 1 second intervals
    for i in range(duration):
        frame_i = int(i * int(fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(dir_name, f'frame_{i}.jpg'), frame)
    cap.release()
    return calibrate_intr(dir_name)

def calibrate_extr(camL, camR, file_pathL, file_pathR):
    """ Calibrate the extrinsic parameters of the camera given 3D-2D point correspondences.
    
    Args:
        cam (Camera): Camera object with known intrinsic parameters.
        file_path (str): Path to the file containing 3D-2D point correspondences.
    Returns:
        (CameraL, CameraR): Camera objects with updated extrinsic parameters.
    """
    # optimize the extrinsic parameters to minimize reprojection error
    imgL = cv2.imread(file_pathL)
    retL, cornersL = cv2.findChessboardCorners(imgL, (7,7), None)
    pts2L = cornersL.squeeze().T
    imgR = cv2.imread(file_pathR)
    retR, cornersR = cv2.findChessboardCorners(imgR, (7,7), None)
    pts2R = cornersR.squeeze().T
    
    # transpose grid for left camera to fix checkerboard detection orientation issue
    pts2L_grid = pts2L.T.reshape(7,7,2)
    pts2L_fixed = np.transpose(pts2L_grid, (1,0,2))
    pts2L = pts2L_fixed.reshape(-1,2).T


    # transpose grid for right camera to fix checkerboard detection orientation issue:
    pts2R_grid = pts2R.T.reshape(7,7,2)
    pts2R_fixed = np.fliplr(pts2R_grid)
    pts2R = pts2R_fixed.reshape(-1, 2).T

    # visualize left camera:
    cv2.drawChessboardCorners(imgL, (7,7), cornersL, retL)
    window_name_L = "Left Camera"
    cv2.namedWindow(window_name_L, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name_L, 800, 600)
    cv2.imshow(window_name_L, imgL)
    cv2.waitKey(1000)

    # visualize right camera:
    cv2.drawChessboardCorners(imgR, (7,7), cornersR, retR)
    
    window_name_R = "Right Camera"
    cv2.namedWindow(window_name_R, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name_R, 800, 600)
    cv2.imshow(window_name_R, imgR)
    cv2.waitKey(1000)

    # create checkerboard 3D points:
    pts3 = np.zeros((3,7*7))
    yy,xx = np.meshgrid(np.arange(7),np.arange(7))
    pts3[0,:] = 2.8*xx.reshape(1,-1)
    pts3[1,:] = 2.8*yy.reshape(1,-1)

    camL = calibratePose(pts3,pts2L,camL,np.array([0,0.2,0,-40,0,-200]))
    camR = calibratePose(pts3,pts2R,camR,np.array([0,-0.2,0,40,0,-200]))

    return camL, camR

if __name__ == '__main__':
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent
    dir_path = base_dir / 'data' / 'checkerboard1'
    camLi, camRi = calibrate_intr(str(dir_path))
    print(camLi)
    camL, camR = calibrate_extr(camLi, camRi, f'{dir_path}/positions/left.JPG', f'{dir_path}/positions/right.JPG')
    print(f'Left camera: {camL}')
    print(f'Right camera: {camR}')


    lookL = np.hstack((camL.t,camL.t+camL.R @ np.array([[0,0,30]]).T))
    lookR = np.hstack((camR.t,camR.t+camR.R @ np.array([[0,0,30]]).T))
    
    # create checkerboard 3D points:
    pts3 = np.zeros((3,7*7))
    yy,xx = np.meshgrid(np.arange(7),np.arange(7))
    pts3[0,:] = 2.8*xx.reshape(1,-1)
    pts3[1,:] = 2.8*yy.reshape(1,-1)

    #Plot the camera positions
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d')
    ax.plot(camR.t[0],camR.t[1],camR.t[2],'ro', label='Right Camera')
    ax.plot(camL.t[0],camL.t[1],camL.t[2],'bo', label='Left Camera')
    ax.plot(lookL[0,:],lookL[1,:],lookL[2,:],'b-', linewidth=2)
    ax.plot(lookR[0,:],lookR[1,:],lookR[2,:],'r-', linewidth=2)
    ax.scatter(pts3[0,:],pts3[1,:],pts3[2,:],c='k',marker='x',label='Checkerboard Points')
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')     
    ax.set_zlabel('Z (cm)')
    ax.set_title("Camera Localization Result")
    ax.legend()

    # set plot viewing angle:
    # z+ into the screen, x+ to the right, y+ up
    ax.view_init(elev=-90, azim=90, roll=0)

    # fix aspect ratio so 1cm is standard in all directions
    visutils.set_axes_equal_3d(ax)
    plt.show()