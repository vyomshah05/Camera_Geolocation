from camutils import Camera, residuals, calibratePose, triangulate
from calibrate_v2 import calibrate
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.spatial.transform import Rotation as R
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import visutils
import cv2
import os
import pickle
import json


def calibratePoseSmooth(pts3, pts2, cam_init, params_init, prev_params=None, smoothness_weight=10.0):
    """ Add temporal smoothness regularization to camera pose calibration
    
    Args:
        pts3
        pts2
        cam_init: Camera object to update
        params_init: Initial parameters [rx, ry, rz, tx, ty, tz]
        prev_params: Previous frame's parameters for smoothness penalty
        smoothness_weight: Weight for smoothness regularization: (higher = smoother)
    Returns:
        Camera: Updated camera object
    """
    if prev_params is None:
        # Use standard cali if no prev frame
        return calibratePose(pts3, pts2, cam_init, params_init)
    
    # error function with smoothness penalty
    def residuals_smooth(params):
        
        reproj_error = residuals(pts3, pts2, cam_init, params)
        
        # Smoothness penalty: penalize deviation from previous parameters
        # deviation between rotation and translation should be weighted seperately
        rotation_diff = params[0:3] - prev_params[0:3]  # in degrees
        translation_diff = params[3:6] - prev_params[3:6]  # in cm
        
        # Scale rotation difference (normalize by expected max change per frame)
        # since our recording only has very slow movement, we can expect < 10 degrees rotation
        smoothness_penalty = np.concatenate([
            smoothness_weight * rotation_diff / 10.0,  # normalize to ~10 deg
            smoothness_weight * translation_diff / 20.0  # normalize to ~20 cm
        ])
        
        return np.concatenate([reproj_error, smoothness_penalty])
    
    # solve least squares with smoothness constraint
    popt, _ = scipy.optimize.leastsq(residuals_smooth, params_init)
    cam_init.update_extrinsics(popt)
    return cam_init


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

def calibrate_intr_from_vid(dir_name, video_path):
    """ Calibrate the camera using images extracted from a video in the specified directory.
    
    Args:
        dir_name (str): Directory containing calibration video.
    Returns:
       (CameraL, CameraR): Calibrated camera object.
    """ 
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(frame_count / fps)
    
    print(f"Video FPS: {fps}, Total Frames: {frame_count}, Duration (s): {duration}")
    #extract frames at 1 second intervals
    for i in range(duration):
        frame_i = int(i * int(fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(dir_name, f'frame_{i}.jpg'), frame)
    cap.release()
    return calibrate_intr(dir_name)

def calibrate_extr_from_vid(dir_name, camL, camR):
    """ Calibrate the extrinsic parameters using images extracted from left and right videos.
    
    Args:
        dir_name (str): Directory containing left.MP4 and right.MP4 videos.
        camL (Camera): Left camera object with known intrinsic parameters.
        camR (Camera): Right camera object with known intrinsic parameters.
    Returns:
        list: List of tuples (camL, camR, pts3) for each frame pair.
    """ 
    # construct paths to left and right videos
    left_video_path = os.path.join(dir_name, 'left.MP4')
    right_video_path = os.path.join(dir_name, 'right.MP4')
    
    # Open both videos
    capL = cv2.VideoCapture(left_video_path)
    capR = cv2.VideoCapture(right_video_path)
    
    # Get video properties for left camera
    fpsL = capL.get(cv2.CAP_PROP_FPS)
    frame_countL = int(capL.get(cv2.CAP_PROP_FRAME_COUNT))
    durationL = frame_countL / fpsL
    
    # get video properties for right camera
    fpsR = capR.get(cv2.CAP_PROP_FPS)
    frame_countR = int(capR.get(cv2.CAP_PROP_FRAME_COUNT))
    durationR = frame_countR / fpsR
    

    # extract frames at _intervals_ second intervals
    intervals = 1

    # compute minimum duration
    duration = min(durationL, durationR)
    num_frames = int(duration / intervals)  # extract every 'intervals' seconds
    
    print(f"Left Video FPS: {fpsL}, Frames: {frame_countL}, Duration (s): {durationL:.2f}")
    print(f"Right Video FPS: {fpsR}, Frames: {frame_countR}, Duration (s): {durationR:.2f}")
    print(f"Minimum duration: {duration:.2f}s, extracting {num_frames} frame pairs")
    
    # create extracted frames directory
    extracted_dir = os.path.join(dir_name, 'extracted')
    if not os.path.exists(extracted_dir):
        os.makedirs(extracted_dir)
    
    
    frame_pairs = []
    for i in range(num_frames):
        time_stamp = i * intervals
        
        # Extract frames
        frame_idx_L = int(time_stamp * fpsL)
        capL.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_L)
        retL, frameL = capL.read()
        frame_idx_R = int(time_stamp * fpsR)
        capR.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_R)
        retR, frameR = capR.read()
        
        if retL and retR:
            left_path = os.path.join(extracted_dir, f'left_frame_{i}.jpg')
            right_path = os.path.join(extracted_dir, f'right_frame_{i}.jpg')
            cv2.imwrite(left_path, frameL)
            cv2.imwrite(right_path, frameR)
            frame_pairs.append((left_path, right_path))
            print(f"Extracted frame pair {i} at {time_stamp:.1f}s")
    
    capL.release()
    capR.release()
    
    # Process each frame pair with calibrate_extr
    results = []
    prev_paramsL = None
    prev_paramsR = None
    
    for i, (left_path, right_path) in enumerate(frame_pairs):
        print(f"\nProcessing frame pair {i}")
        try:
            # perform deep copy
            import copy
            camL_copy = copy.deepcopy(camL)
            camR_copy = copy.deepcopy(camR)
            
            # Use previous frame's result as initial guess and smoothness constraint
            if prev_paramsL is not None and prev_paramsR is not None:
                init_poseL = prev_paramsL
                init_poseR = prev_paramsR
                print(f"Previous cam params:")
                print(f"Left: {init_poseL}")
                print(f"Right: {init_poseR}")
                camL_cal, camR_cal, pts3 = calibrate_extr(camL_copy, camR_copy, left_path, right_path, init_poseL, init_poseR, prev_paramsL, prev_paramsR)
            else:
                print(f"Use default init params")
                camL_cal, camR_cal, pts3 = calibrate_extr(camL_copy, camR_copy, left_path, right_path)
            
            # Extract parameters from calibrated cameras for next frame
            rot_L = R.from_matrix(camL_cal.R)
            rot_R = R.from_matrix(camR_cal.R)
            euler_L = rot_L.as_euler('xyz', degrees=True)
            euler_R = rot_R.as_euler('xyz', degrees=True)
            prev_paramsL = np.concatenate([euler_L, camL_cal.t.flatten()])
            prev_paramsR = np.concatenate([euler_R, camR_cal.t.flatten()])
            
            print(f"Positions: Left: {camL_cal.t.T}, Right: {camR_cal.t.T}")
            results.append((camL_cal, camR_cal, pts3))
        except Exception as e:
            print(f"Unable to process frame pair {i}: {e}")
            continue
    
    if not results:
        print("None processed")
        return []
    
    # Create interactive visualization with slider
    fig = plt.figure(figsize=(14, 11))
    
    # Adjust figure layout to make room for slider
    plt.subplots_adjust(bottom=0.1, top=0.95, left=0.05, right=0.95, hspace=0.3, wspace=0.3)
    
    # Create subplots
    ax_3d = fig.add_subplot(2, 2, 1, projection='3d')
    ax_xz = fig.add_subplot(2, 2, 2)
    ax_yz = fig.add_subplot(2, 2, 3)
    ax_xy = fig.add_subplot(2, 2, 4)
    
    # Function to update the plot for a given frame index
    def update_plot(frame_idx):
        frame_idx = int(frame_idx)
        camL_cal, camR_cal, pts3 = results[frame_idx]
        
        lookL_short = np.hstack((camL_cal.t, camL_cal.t + camL_cal.R @ np.array([[0,0,50]]).T))
        lookR_short = np.hstack((camR_cal.t, camR_cal.t + camR_cal.R @ np.array([[0,0,50]]).T))
        
        # Clear all axes
        ax_3d.clear()
        ax_xz.clear()
        ax_yz.clear()
        ax_xy.clear()
        
        # Update title
        fig.suptitle(f'Frame {frame_idx} - Camera Calibration (t={frame_idx*intervals:.1f}s)', 
                     fontsize=14, fontweight='bold')
        
        # 3D view
        ax_3d.view_init(elev=-58, azim=51, roll=43)
        ax_3d.scatter(pts3[0,:],pts3[1,:],pts3[2,:],'k', marker='x', label='Checkerboard')
        ax_3d.plot(camR_cal.t[0],camR_cal.t[1],camR_cal.t[2],'ro', markersize=10, label='Right Camera')
        ax_3d.plot(camL_cal.t[0],camL_cal.t[1],camL_cal.t[2],'bo', markersize=10, label='Left Camera')
        ax_3d.plot(lookL_short[0,:],lookL_short[1,:],lookL_short[2,:],'b-', linewidth=2)
        ax_3d.plot(lookR_short[0,:],lookR_short[1,:],lookR_short[2,:],'r-', linewidth=2)
        visutils.set_axes_equal_3d(ax_3d)
        visutils.label_axes(ax_3d)
        ax_3d.set_title('Scene 3D View')
        ax_3d.legend()
        
        # XZ view
        ax_xz.plot(pts3[0,:],-pts3[2,:],'k.', label='Checkerboard')
        ax_xz.plot(camR_cal.t[0],-camR_cal.t[2],'ro', markersize=10, label='Right Camera')
        ax_xz.plot(camL_cal.t[0],-camL_cal.t[2],'bo', markersize=10, label='Left Camera')
        ax_xz.plot(lookL_short[0,:],-lookL_short[2,:],'b-', linewidth=2)
        ax_xz.plot(lookR_short[0,:],-lookR_short[2,:],'r-', linewidth=2)
        ax_xz.set_title('XZ-view (Top View)')
        ax_xz.grid()
        ax_xz.set_xlabel('x (cm)')
        ax_xz.set_ylabel('z (cm)')
        ax_xz.legend()
        ax_xz.axis('equal')
        
        # YZ view
        ax_yz.plot(-pts3[2,:],pts3[1,:],'k.', label='Checkerboard')
        ax_yz.plot(-camR_cal.t[2],camR_cal.t[1],'ro', markersize=10, label='Right Camera')
        ax_yz.plot(-camL_cal.t[2],camL_cal.t[1],'bo', markersize=10, label='Left Camera')
        ax_yz.plot(-lookL_short[2,:],lookL_short[1,:],'b-', linewidth=2)
        ax_yz.plot(-lookR_short[2,:],lookR_short[1,:],'r-', linewidth=2)
        ax_yz.set_title('YZ-view (Side View)')
        ax_yz.grid()
        ax_yz.set_xlabel('-z (cm)')
        ax_yz.set_ylabel('y (cm)')
        ax_yz.legend()
        ax_yz.axis('equal')
        
        # XY view
        ax_xy.plot(pts3[0,:],pts3[1,:],'k.', label='Checkerboard')
        ax_xy.plot(camR_cal.t[0],camR_cal.t[1],'ro', markersize=10, label='Right Camera')
        ax_xy.plot(camL_cal.t[0],camL_cal.t[1],'bo', markersize=10, label='Left Camera')
        ax_xy.plot(lookL_short[0,:],lookL_short[1,:],'b-', linewidth=2)
        ax_xy.plot(lookR_short[0,:],lookR_short[1,:],'r-', linewidth=2)
        ax_xy.set_title('XY-view (Front View from Checkerboard)')
        ax_xy.grid()
        ax_xy.set_xlabel('x (cm)')
        ax_xy.set_ylabel('y (cm)')
        ax_xy.legend()
        ax_xy.axis('equal')
        
        fig.canvas.draw_idle()
    

    update_plot(0)
    
    # slider
    ax_slider = plt.axes([0.15, 0.02, 0.7, 0.03])
    slider = Slider(
        ax=ax_slider,
        label='Frame',
        valmin=0,
        valmax=len(results) - 1,
        valinit=0,
        valstep=1
    )
    
    # update
    slider.on_changed(update_plot)
    
    # Save camera positions to JSON file
    camera_positions = []
    for i, (camL_cal, camR_cal, pts3) in enumerate(results):
        camera_positions.append({
            'frame': i,
            'timestamp_seconds': i * intervals,
            'left_camera_position': {
                'x': float(camL_cal.t[0, 0]),
                'y': float(camL_cal.t[1, 0]),
                'z': float(camL_cal.t[2, 0])
            },
            'right_camera_position': {
                'x': float(camR_cal.t[0, 0]),
                'y': float(camR_cal.t[1, 0]),
                'z': float(camR_cal.t[2, 0])
            }
        })
    
    json_path = os.path.join(dir_name, 'camera_positions.json')
    with open(json_path, 'w') as f:
        json.dump(camera_positions, f, indent=2)
    print(f"\ncam pos dir: {json_path}")
    
    plt.show()
    
    return results

def calibrate_extr(camL, camR, file_pathL, file_pathR, init_poseL=None, init_poseR=None, prev_poseL=None, prev_poseR=None):
    """ Calibrate the extrinsic parameters of the camera given 3D-2D point correspondences.
    
    Args:
        camL (Camera): Left camera object with known intrinsic parameters.
        camR (Camera): Right camera object with known intrinsic parameters.
        file_pathL (str): Path to the left image file.
        ...
    Returns:
        (CameraL, CameraR, pts3): Camera objects with updated extrinsic parameters and triangulated points.
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

    # Initial pose parameters: [rx, ry, rz, tx, ty, tz]
    # Use provided initial poses or default values
    if init_poseL is None:
        init_poseL = np.array([-180, 0, 0, -145, -61, 208])
    if init_poseR is None:
        init_poseR = np.array([-180, 0, 0, 135, 90, 358])
    

    camL = calibratePoseSmooth(pts3, pts2L, camL, init_poseL, prev_poseL, smoothness_weight=50.0)
    camR = calibratePoseSmooth(pts3, pts2R, camR, init_poseR, prev_poseR, smoothness_weight=50.0)

    pts3 = triangulate(pts2L, camL, pts2R, camR)
    return camL, camR, pts3

if __name__ == '__main__':
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent
    
    # Calibrate intrinsic parameters from video
    dir_path = base_dir / 'data' / 'checkerboard2' / 'calibrate'
    camLi, camRi = calibrate_intr_from_vid(f'{dir_path}/pics_from_vid', f'{dir_path}/video/calibrate.MP4')
    print(camLi)
    
    # Calibrate extrinsic parameters from videos
    videos_dir = base_dir / 'data' / 'checkerboard2' / 'videos'
    results = calibrate_extr_from_vid(str(videos_dir), camLi, camRi)
    
    print(f"\nProcessed {len(results)} frame pairs successfully")
    for i, (camL, camR, pts3) in enumerate(results):
        print(f'\nFrame {i} (t={i*0.5:.1f}s):')
        print(f'  Left camera position: {camL.t.T}')
        print(f'  Right camera position: {camR.t.T}')


    lookL = np.hstack((camL.t,camL.t+camL.R @ np.array([[0,0,30]]).T))
    lookR = np.hstack((camR.t,camR.t+camR.R @ np.array([[0,0,30]]).T))