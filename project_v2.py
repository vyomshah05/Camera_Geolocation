from camutils import Camera, residuals, calibratePose, triangulate
from calibrate_v2 import calibrate
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
    camL = Camera(f,c,R_init,t_init)
    camR = Camera(f,c,R_init,t_init)

    return (camL, camR)


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
    retL, cornersL = cv2.findChessboardCorners(imgL, (8,6), None)
    pts2L = cornersL.squeeze().T
    imgR = cv2.imread(file_pathR)
    retR, cornersR = cv2.findChessboardCorners(imgR, (8,6), None)
    pts2R = cornersR.squeeze().T




    # visualize left camera:
    cv2.drawChessboardCorners(imgL, (8,6), cornersL, retL)
    window_name_L = "Left Camera"
    cv2.namedWindow(window_name_L, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name_L, 800, 600)
    cv2.imshow(window_name_L, imgL)
    cv2.waitKey(1000)

    # visualize right camera:
    cv2.drawChessboardCorners(imgR, (8,6), cornersR, retR)
    
    window_name_R = "Right Camera"
    cv2.namedWindow(window_name_R, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name_R, 800, 600)
    cv2.imshow(window_name_R, imgR)
    cv2.waitKey(1000)

    # create checkerboard 3D points:
    pts3 = np.zeros((3,8*6))
    xx,yy = np.meshgrid(np.arange(8),np.arange(6))
    pts3[0,:] = 2.8*xx.reshape(1,-1) 
    pts3[1,:] = 2.8*yy.reshape(1,-1)
    # pose parameters: [rx, ry, rz, tx, ty, tz]
    # change x rotation so camera is looking at checkerboard (Camera's Z axis points from checkerboard outwards)
    # changed so left is low and close, while right is high and far
    camL = calibratePose(pts3,pts2L,camL,np.array([-180, 0, 0, -145, -61, 208]))
    camR = calibratePose(pts3,pts2R,camR,np.array([-180, 0, 0, 135, 90, 358]))

    pts3 = triangulate(pts2L, camL, pts2R, camR)
    return camL, camR, pts3
    

if __name__ == '__main__':
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent
    dir_path = base_dir / 'data' / 'checkerboard2' / 'calibrate' / '1'
    camLi, camRi = calibrate_intr(str(dir_path))
    print(camLi)
    dir_path = base_dir / 'data' / 'checkerboard2'
    camL, camR, pts3 = calibrate_extr(camLi, camRi, f'{dir_path}/positions1/left.JPG', f'{dir_path}/positions1/right.JPG')
    print(f'Left camera: {camL}')
    print(f'Right camera: {camR}')


    lookL = np.hstack((camL.t,camL.t+camL.R @ np.array([[0,0,30]]).T))
    lookR = np.hstack((camR.t,camR.t+camR.R @ np.array([[0,0,30]]).T))

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
    # z+ out of the screen, x+ to the right, y+ up
    ax.view_init(elev=35, azim=163, roll=-100)

    # fix aspect ratio so 1cm is standard in all directions
    visutils.set_axes_equal_3d(ax)
    
    # Multi-view visualization
    lookL_short = np.hstack((camL.t,camL.t+camL.R @ np.array([[0,0,50]]).T))
    lookR_short = np.hstack((camR.t,camR.t+camR.R @ np.array([[0,0,50]]).T))
    
    # visualize the left and right cameras from multiple views
    fig2 = plt.figure(figsize=(12, 10))
    
    # 3D view
    ax = fig2.add_subplot(2,2,1,projection='3d')
    ax.view_init(elev=35, azim=163, roll=-100)
    ax.scatter(pts3[0,:],pts3[1,:],pts3[2,:],'k', marker='x', label='Checkerboard')
    ax.plot(camR.t[0],camR.t[1],camR.t[2],'ro', markersize=10, label='Right Camera')
    ax.plot(camL.t[0],camL.t[1],camL.t[2],'bo', markersize=10, label='Left Camera')
    ax.plot(lookL_short[0,:],lookL_short[1,:],lookL_short[2,:],'b-', linewidth=2)
    ax.plot(lookR_short[0,:],lookR_short[1,:],lookR_short[2,:],'r-', linewidth=2)
    visutils.set_axes_equal_3d(ax)
    visutils.label_axes(ax)
    ax.set_title('Scene 3D View')
    ax.legend()
    
    # XZ view
    ax = fig2.add_subplot(2,2,2)
    ax.plot(pts3[0,:],-pts3[2,:],'k.', label='Checkerboard')
    ax.plot(camR.t[0],-camR.t[2],'ro', markersize=10, label='Right Camera')
    ax.plot(camL.t[0],-camL.t[2],'bo', markersize=10, label='Left Camera')
    ax.plot(lookL_short[0,:],-lookL_short[2,:],'b-', linewidth=2)
    ax.plot(lookR_short[0,:],-lookR_short[2,:],'r-', linewidth=2)
    ax.set_title('XZ-view (Top View)')
    ax.grid()
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('z (cm)')
    ax.legend()
    ax.axis('equal')
    
    # YZ view (rotated 90° counterclockwise)
    ax = fig2.add_subplot(2,2,3)
    ax.plot(-pts3[2,:],pts3[1,:],'k.', label='Checkerboard')
    ax.plot(-camR.t[2],camR.t[1],'ro', markersize=10, label='Right Camera')
    ax.plot(-camL.t[2],camL.t[1],'bo', markersize=10, label='Left Camera')
    ax.plot(-lookL_short[2,:],lookL_short[1,:],'b-', linewidth=2)
    ax.plot(-lookR_short[2,:],lookR_short[1,:],'r-', linewidth=2)
    ax.set_title('YZ-view (Side View)')
    ax.grid()
    ax.set_xlabel('-z (cm)')
    ax.set_ylabel('y (cm)')
    ax.legend()
    ax.axis('equal')
    
    # XY view
    ax = fig2.add_subplot(2,2,4)
    ax.plot(pts3[0,:],pts3[1,:],'k.', label='Checkerboard')
    ax.plot(camR.t[0],camR.t[1],'ro', markersize=10, label='Right Camera')
    ax.plot(camL.t[0],camL.t[1],'bo', markersize=10, label='Left Camera')
    ax.plot(lookL_short[0,:],lookL_short[1,:],'b-', linewidth=2)
    ax.plot(lookR_short[0,:],lookR_short[1,:],'r-', linewidth=2)
    ax.set_title('XY-view (Front View from Checkerboard)')
    ax.grid()
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.legend()
    ax.axis('equal')
    
    plt.tight_layout()
    plt.show()