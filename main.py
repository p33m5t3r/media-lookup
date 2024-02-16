import cv2
from PIL import Image
import torch
import open_clip
import os
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from dataclasses import dataclass
from typing import Any

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

CLIP_CFG = get_clip_cfg()

def embed_img(img: Image, clip_cfg: ClipConfig):
    img = img.convert('RGB')
    img = clip_cfg.transform(img).unsqueeze(0).to(clip_cfg.device)
    n = 20  # truncate/pad to shape (1xn)
    with torch.no_grad(), torch.cuda.amp.autocast():
        t = clip_cfg.model.generate(img)[0][:n]
        p = torch.zeros(n)
        p[:t.shape[0]] = t

    # print(features[0])
    # print(features.shape)
    return p 

class ClipEmbeddingFn(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return [embed_img(img, CLIP_CFG) for img in input]

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

def index_video(dir_name, filename, coll):
    frames = extract_frames(dir_name, filename)
    ids = [frame[0] for frame in frames]
    # this is all just inefficient hack garbage to get around chromadb
    # will replace with something else later
    chroma_hack = lambda t: t.cpu().numpy().tolist()
    embeddings = [chroma_hack(embed_img(frame[1], CLIP_CFG)) for frame in frames]
    coll.add(
            documents=ids,
            ids=ids,
            embeddings=embeddings,
        )

    print(f'Indexed {filename}')

def index_dir(dir_name: str, coll):
    for filename in os.listdir(dir_name):
        if filename.endswith('.mp4'):
            print(f'Indexing {filename}')
            index_video(dir_name, filename, coll)


def emb_query(q: str):
    return CLIP_CFG.tokenizer(q).cpu().numpy().tolist()[0][:20]

if __name__ == '__main__':
    
    # setup / config
    client = chromadb.Client()
    embed_fn = ClipEmbeddingFn()
    collection = client.create_collection(
        name="video_frames",
        embedding_function=embed_fn,
    )
    MEDIA_ROOT = 'media/'

    # index video frames

    print(CLIP_CFG.tokenizer("An acoustic guitar.").shape)

    index_dir(MEDIA_ROOT, collection)

    q = "An acoustic guitar."
    # print(emb_query(q))
    res = collection.query(
        query_embeddings=emb_query(q),
        include=['documents']
    )

    print(res)


    """
    MEDIA_ROOT = 'media/'
    filename = 'video.mp4'
    extract_frames(MEDIA_ROOT, filename, sample_rate=1)
    
    print(open_clip.decode(features[0]).split("<end_of_text>")[0].replace("<start_of_text>", ""))
    print(features)
    print(features.shape)

    test_txt = "An acoustic guitar."
    print(tokenizer(test_txt).to(device))
    # image_features /= image_features.norm(dim=-1, keepdim=True)
    """


