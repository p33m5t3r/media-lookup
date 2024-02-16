import cv2
import numpy as np

def extract_frames(video_path, sample_rate=1):
    """
    Extracts frames from a video file.

    :param video_path: Path to the video file.
    :param sample_rate: Number of seconds between frames to sample.
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame if it's on the specified sample rate
        if frame_count % (fps * sample_rate) == 0:
            # Save frame as PNG file
            cv2.imwrite(f'frame_{frame_count}.png', frame)
            print(f'Extracted frame {frame_count}')

        frame_count += 1

    cap.release()

# Example usage
video_path = 'video.mp4'
extract_frames(video_path, sample_rate=1)  # Adjust sample_rate as needed

