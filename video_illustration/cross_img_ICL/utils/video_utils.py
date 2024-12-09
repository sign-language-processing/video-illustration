import cv2
import os

def extract_frames(video_path, output_folder):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    
    if not video_capture.isOpened():
        print("Error: Could not open video.")
        return
    
    os.makedirs(output_folder, exist_ok=True)
    
    frame_number = 0
    
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        frame_filename = os.path.join(output_folder, f"frame_{frame_number:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        print(f"Saved: {frame_filename}")
        frame_number += 1
    
    # Release the video capture object
    video_capture.release()
    print(f"Extraction complete. {frame_number} frames extracted.")


if  __name__ == '__main__':

    video_file_path = "/Users/jannabruner/Documents/research/sign_language_project/SLP_illistration/sample_3/bonn.mp4"
    output_directory = "/Users/jannabruner/Documents/research/sign_language_project/SLP_illistration/sample_3/frames"
    extract_frames(video_file_path, output_directory)
