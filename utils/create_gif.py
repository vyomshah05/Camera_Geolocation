from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path
import cv2
import numpy as np

def create_camera_gif(extracted_dir, output_path='camera_animation.gif', duration=500, font_size=40, show_checkerboard=True):
    """
    Create a GIF from extracted camera frames with labels.
    
    Args:
        extracted_dir: Directory containing left_frame_X.jpg and right_frame_X.jpg files
        output_path: Output GIF file path
        duration: Duration of each frame in milliseconds
        font_size: Size of the label text
    """
    # Find all frame pairs
    extracted_path = Path(extracted_dir)
    def frame_index(p: Path) -> int:
    # assumes names like left_frame_123.jpg
        return int(p.stem.split('_')[-1])

    left_frames  = sorted(extracted_path.glob('left_frame_*.jpg'),  key=frame_index)
    right_frames = sorted(extracted_path.glob('right_frame_*.jpg'), key=frame_index)
    
    if not left_frames or not right_frames:
        print(f"No frames found in {extracted_dir}")
        return
    
    print(f"Found {len(left_frames)} left frames and {len(right_frames)} right frames")
    
    # Ensure we have matching pairs
    num_frames = min(len(left_frames), len(right_frames))
    
    combined_frames = []
    
    for i in range(num_frames):
        # Load images with OpenCV for checkerboard detection
        left_img_cv = cv2.imread(str(left_frames[i]))
        right_img_cv = cv2.imread(str(right_frames[i]))
        
        # Detect and draw checkerboard corners if enabled
        if show_checkerboard:
            # Detect checkerboard in left image
            ret_left, corners_left = cv2.findChessboardCorners(left_img_cv, (8, 6), None)
            
            # Detect checkerboard in right image
            ret_right, corners_right = cv2.findChessboardCorners(right_img_cv, (8, 6), None)
            
            # Only draw if BOTH cameras detect the checkerboard successfully
            if ret_left and ret_right:
                cv2.drawChessboardCorners(left_img_cv, (8, 6), corners_left, ret_left)
                cv2.drawChessboardCorners(right_img_cv, (8, 6), corners_right, ret_right)
            else:
                # Add warning text if detection failed
                if not ret_left:
                    cv2.putText(left_img_cv, "NO CHECKERBOARD DETECTED", (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if not ret_right:
                    cv2.putText(right_img_cv, "NO CHECKERBOARD DETECTED", (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Convert from BGR to RGB for PIL
        left_img_rgb = cv2.cvtColor(left_img_cv, cv2.COLOR_BGR2RGB)
        right_img_rgb = cv2.cvtColor(right_img_cv, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Images
        left_img = Image.fromarray(left_img_rgb)
        right_img = Image.fromarray(right_img_rgb)
        
        # Get dimensions
        width, height = left_img.size
        
        # Create combined image: right camera on left, left camera on right
        combined_width = width * 2
        combined_height = height + 60  # Extra space for labels at top
        
        combined = Image.new('RGB', (combined_width, combined_height), color='white')
        
        # Paste images (right camera on left side, left camera on right side)
        combined.paste(right_img, (0, 60))
        combined.paste(left_img, (width, 60))
        
        # Add labels
        draw = ImageDraw.Draw(combined)
        
        try:
            # Try to use a nice font
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            # Fall back to default font
            font = ImageFont.load_default()
        
        # Add text labels
        label_right = "Camera Right"
        label_left = "Camera Left"
        
        # Calculate text positions (centered above each image)
        bbox_right = draw.textbbox((0, 0), label_right, font=font)
        bbox_left = draw.textbbox((0, 0), label_left, font=font)
        
        text_width_right = bbox_right[2] - bbox_right[0]
        text_width_left = bbox_left[2] - bbox_left[0]
        
        x_right = (width - text_width_right) // 2
        x_left = width + (width - text_width_left) // 2
        
        draw.text((x_right, 10), label_right, fill='red', font=font)
        draw.text((x_left, 10), label_left, fill='blue', font=font)
        
        # Add frame number
        frame_text = f"Frame {i}"
        draw.text((10, combined_height - 30), frame_text, fill='black', font=font)
        
        combined_frames.append(combined)
        print(f"Processed frame {i}/{num_frames}")
    
    # Save as GIF
    if combined_frames:
        combined_frames[0].save(
            output_path,
            save_all=True,
            append_images=combined_frames[1:],
            duration=duration,
            loop=0
        )
        print(f"\nGIF saved to: {output_path}")
        print(f"Total frames: {len(combined_frames)}")
        print(f"Duration per frame: {duration}ms")
    else:
        print("No frames to save")

if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent
    extracted_dir = base_dir / 'data' / 'checkerboard2' / 'videos' / 'extracted'
    output_file = base_dir / 'data' / 'checkerboard2' / 'videos' / 'camera_animation.gif'
    
    create_camera_gif(
        extracted_dir=str(extracted_dir),
        output_path=str(output_file),
        duration=100,  # 100ms per frame (10 fps)
        font_size=40,
        show_checkerboard=True  # Enable checkerboard corner visualization
    )
