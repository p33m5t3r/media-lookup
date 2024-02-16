import cv2
from PIL import Image
import torch
import open_clip
import os
import chromadb
from dataclasses import dataclass
from typing import Any
import sys

# fuck chromadb, all my homies hate chromadb. just diy it bro.
# but i'm lazy, so i'll use it for now. 
@dataclass
class ClipConfig:
    device: str
    model: open_clip.model
    transform: Any
    tokenizer: Any

def get_clip_cfg():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, transform = open_clip.create_model_and_transforms(
          model_name="coca_ViT-L-14",
          pretrained="mscoco_finetuned_laion2B-s13B-b90k",
          device=device
    )
    tokenizer = open_clip.get_tokenizer('coca_ViT-L-14')

    return ClipConfig(device, model, transform, tokenizer)

def embed_img(img: Image, clip_cfg: ClipConfig):
    img = img.convert('RGB')
    img = clip_cfg.transform(img).unsqueeze(0).to(clip_cfg.device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        return clip_cfg.model.generate(img)

def extract_frames(_dir, filename, sample_rate=1):
    frames = []
    cap = cv2.VideoCapture(f'{_dir}{filename}')
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
    frame_count = 0
    thumb_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame if it's on the specified sample rate
        if frame_count % (fps * sample_rate) == 0:
            thumb_count += 1
            # Save frame as PNG file
            # cv2.imwrite(f'{_dir}frame_{frame_count}.png', frame)
            timestamp = frame_count / fps
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
            frames.append((f"{_dir}{filename} @t={timestamp}s", img_pil))

        frame_count += 1

    cap.release()
    print(f'Extracted {thumb_count} frames from {filename}')
    return frames

def clip_decode(features):
    return open_clip.decode(features[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")

def index_video(dir_name, filename, coll, ccfg):
    frames = extract_frames(dir_name, filename)
    ids = [frame[0] for frame in frames]
    embeddings = [embed_img(frame[1], ccfg) for frame in frames]
    docs = [clip_decode(embed) for embed in embeddings]
    coll.add(
            documents=docs,
            ids=ids,
        )

    print(f'Indexed {filename}')

def index_dir(dir_name: str, coll, ccfg):
    for filename in os.listdir(dir_name):
        if filename.endswith('.mp4'):
            print(f'Indexing {filename}')
            index_video(dir_name, filename, coll, ccfg)


def emb_query(q: str, ccfg):
    return ccfg.tokenizer(q).cpu().numpy().tolist()[0][:20]


def index(media_root, collection):
    print("loading model...")
    CLIP_CFG = get_clip_cfg()
    index_dir(media_root, collection, CLIP_CFG)

def query(q, collection):
    res = collection.query(
        query_texts=[q],
        n_results=1,
    )

    return res['ids'][0][0]


# https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_coca.ipynb
if __name__ == '__main__':
        # setup / config
    VDB_PATH = 'vectordb'
    MEDIA_ROOT = 'media/'

    client = chromadb.PersistentClient(path=VDB_PATH)
    collection = client.get_or_create_collection(
        name="video_frames",
    )

    args = sys.argv[1:]
    if len(args) >= 1:
        if args[0] == 'index':
            index(MEDIA_ROOT, collection)
        elif args[0] == 'find':
            q = args[1]
            exit(0)
            print(f'searching for: \"{q}\"')

            n_results = 3
            res = collection.query(
                query_texts=[q],
                n_results=n_results,
            )
            print("\nresults:")
            for r in range(n_results):
                _id = res['ids'][0][r]
                _doc = res['documents'][0][r]
                print(f'  {_id} - \"{_doc}\"')

