import json
import numpy as np
from pathlib import Path


def calculate_relative_position_error(left_est, right_est, left_gt, right_gt):
    estimated_distance = np.linalg.norm(right_est - left_est)
    ground_truth_distance = np.linalg.norm(right_gt - left_gt)
    
    return abs(estimated_distance - ground_truth_distance)


def validate_camera_positions(json_path, video_duration_left=19.0, video_duration_right=19.0):
    """
    Validate camera positions from JSON against ground truth.
    
    Args:
        json_path: Path to camera_positions.json
        video_duration_left: Total duration of left video (unused now)
        video_duration_right: Total duration of right video (unused now)
    
    Returns:
        None
    """
    # Load camera positions
    with open(json_path, 'r') as f:
        camera_data = json.load(f)
    
    print("Running camera position validation calculations:")
    
    for frame_data in camera_data:
        frame = frame_data['frame']
        
        # Only validate start (frame 0) and end (last frame)
        if frame != 0 and frame != len(camera_data) - 1:
            continue
        
        left_est = np.array([
            frame_data['left_camera_position']['x'],
            frame_data['left_camera_position']['y'],
            frame_data['left_camera_position']['z']
        ])
        
        right_est = np.array([
            frame_data['right_camera_position']['x'],
            frame_data['right_camera_position']['y'],
            frame_data['right_camera_position']['z']
        ])
        
        # ground truth positions (measured manually with tape measure)
        if frame == 0:
            left_gt = np.array([-135, -61, -208])
            right_gt = np.array([145, 90, -358])
            label = "Start"
        else: 
            left_gt = np.array([-40, -24, -40])
            right_gt = np.array([-135, -61, -208])
            label = "End"
        
  
        left_error = np.linalg.norm(left_est - left_gt)
        right_error = np.linalg.norm(right_est - right_gt)
        relative_error = calculate_relative_position_error(left_est, right_est, left_gt, right_gt)
    
        left_gt_distance = np.linalg.norm(left_gt)
        right_gt_distance = np.linalg.norm(right_gt)
        gt_relative_distance = np.linalg.norm(right_gt - left_gt)
        
        if left_gt_distance>0:
            left_pct = (left_error / left_gt_distance) * 100
        else:
            left_pct = 0
        
        if right_gt_distance>0:
            right_pct = (right_error / right_gt_distance) * 100
        else:
            right_pct = 0

        if gt_relative_distance > 0:
            relative_pct = (relative_error / gt_relative_distance) * 100
        else:
            relative_pct = 0
        

        if frame == 0:
            print("\nLeft Camera Positioning Error:")
        print(f"  {label}:")
        print(f"    Estimated t:    [{left_est[0]:.2f}, {left_est[1]:.2f}, {left_est[2]:.2f}]")
        print(f"    Ground Truth t: [{left_gt[0]:.2f}, {left_gt[1]:.2f}, {left_gt[2]:.2f}]")
        print(f"    Error:          {left_error:.2f} cm ({left_pct:.2f}%)")
        
        if frame == len(camera_data) - 1:
            print("\nRight Camera Positioning Error:")
            right_start_est = np.array([camera_data[0]['right_camera_position']['x'],
                                       camera_data[0]['right_camera_position']['y'],
                                       camera_data[0]['right_camera_position']['z']])
            right_start_gt = np.array([145, 90, -358])
            right_start_error = np.linalg.norm(right_start_est - right_start_gt)
            right_start_pct = (right_start_error / np.linalg.norm(right_start_gt)) * 100
            
            print(f"  Start:")
            print(f"    Estimated t:     [{right_start_est[0]:.2f}, {right_start_est[1]:.2f}, {right_start_est[2]:.2f}]")
            print(f"    Ground Truth t:  [{right_start_gt[0]:.2f}, {right_start_gt[1]:.2f}, {right_start_gt[2]:.2f}]")
            print(f"    Error:           {right_start_error:.2f} cm ({right_start_pct:.2f}%)")
            print(f"  End:")
            print(f"    Estimated t:     [{right_est[0]:.2f}, {right_est[1]:.2f}, {right_est[2]:.2f}]")
            print(f"    Ground Truth t:  [{right_gt[0]:.2f}, {right_gt[1]:.2f}, {right_gt[2]:.2f}]")
            print(f"    Error:           {right_error:.2f} cm ({right_pct:.2f}%)")
            
            left_start_est = np.array([camera_data[0]['left_camera_position']['x'],
                                      camera_data[0]['left_camera_position']['y'],
                                      camera_data[0]['left_camera_position']['z']])
            left_start_gt = np.array([-135, -61, -208])
            left_start_error = np.linalg.norm(left_start_est - left_start_gt)
            left_start_pct = (left_start_error / np.linalg.norm(left_start_gt)) * 100
            
            rel_start_error = calculate_relative_position_error(left_start_est, right_start_est, left_start_gt, right_start_gt)
            rel_start_pct = (rel_start_error / np.linalg.norm(right_start_gt - left_start_gt)) * 100
            
            print("\nRelative (Camera-to-Camera Distance) Positioning:")
            print(f"  Start Error:      {rel_start_error:.2f} cm ({rel_start_pct:.2f}%)")
            print(f"  End Error:        {relative_error:.2f} cm ({relative_pct:.2f}%)")
            
            print("\nAverage of left and right camera absolute positioning errors:")
            overall_mean = (left_start_error + left_error + right_start_error + right_error) / 4
            overall_pct = (left_start_pct + left_pct + right_start_pct + right_pct) / 4
            print(f"  Mean Absolute Error: {overall_mean:.2f} cm")
            print(f"  Mean Percentage Error:{overall_pct:.2f}%")



if __name__ == '__main__':
    # Test the validation
    base_dir = Path(__file__).resolve().parent
    json_path = base_dir / 'data' / 'checkerboard2' / 'videos' / 'camera_positions.json'
    
    if json_path.exists():
        results = validate_camera_positions(str(json_path))
    else:
        print(f"Error: {json_path} not found")
