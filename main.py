import cv2
from PIL import Image
import torch
import open_clip
# import numpy as np

def extract_frames(_dir, filename, sample_rate=1):
    """
    Extracts frames from a video file.

    :param video_path: Path to the video file.
    :param sample_rate: Number of seconds between frames to sample.
    """
    cap = cv2.VideoCapture(f'{_dir}{filename}')
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame if it's on the specified sample rate
        if frame_count % (fps * sample_rate) == 0:
            # Save frame as PNG file
            cv2.imwrite(f'{_dir}frame_{frame_count}.png', frame)
            print(f'Extracted frame {frame_count}')

        frame_count += 1

    cap.release()


if __name__ == '__main__':

    """
    MEDIA_ROOT = 'media/'
    filename = 'video.mp4'
    extract_frames(MEDIA_ROOT, filename, sample_rate=1)
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, transform = open_clip.create_model_and_transforms(
          model_name="coca_ViT-L-14",
          pretrained="mscoco_finetuned_laion2B-s13B-b90k",
          device=device
    )
    tokenizer = open_clip.get_tokenizer('coca_ViT-L-14')

    im = Image.open('media/frame_120.png').convert('RGB')
    im = transform(im).unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        features = model.generate(im)

    print(open_clip.decode(features[0]).split("<end_of_text>")[0].replace("<start_of_text>", ""))
    print(features)
    print(features.shape)

    test_txt = "An acoustic guitar."
    print(tokenizer(test_txt).to(device))
    # image_features /= image_features.norm(dim=-1, keepdim=True)


