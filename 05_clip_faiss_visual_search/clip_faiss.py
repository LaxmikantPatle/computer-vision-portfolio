import torch
import open_clip
import cv2
import numpy as np
from PIL import Image
import faiss
import pickle
import json
from pathlib import Path
from typing import List, Tuple

MODEL = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


class ClipIndexer:
    def __init__(self, device="cpu"):
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED, device=self.device)
        self.tokenizer = open_clip.get_tokenizer(MODEL)
        self.index = None
        self.paths = []
        self.embs = None

    @torch.no_grad()
    def embed_image(self, img_path: str) -> np.ndarray:
        img = Image.open(img_path).convert("RGB")
        img_t = self.preprocess(img).unsqueeze(0).to(self.device)
        feats = self.model.encode_image(img_t)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype("float32")

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        txt = self.tokenizer([text]).to(self.device)
        feats = self.model.encode_text(txt)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype("float32")

    def build_index(self, image_dir: str, out_prefix="index"):
        paths = []
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            paths.extend(Path(image_dir).rglob("*" + ext))
        paths = sorted(set(paths))
        if len(paths) == 0:
            raise ValueError(f"no images found in {image_dir}")
        embs = []
        for p in paths:
            try:
                e = self.embed_image(str(p))
                embs.append(e[0])
            except Exception as e:
                print("skip", p, e)
        embs = np.vstack(embs).astype("float32")
        d = embs.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embs)
        self.index = index
        self.paths = [str(p) for p in paths]
        self.embs = embs
        faiss.write_index(index, f"{out_prefix}.faiss")
        with open(f"{out_prefix}_paths.pkl", "wb") as f:
            pickle.dump(self.paths, f)
        print(f"indexed {len(self.paths)} images. index saved to {out_prefix}.faiss")

    def load_index(self, prefix="index"):
        self.index = faiss.read_index(f"{prefix}.faiss")
        with open(f"{prefix}_paths.pkl", "rb") as f:
            self.paths = pickle.load(f)

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        if self.index is None:
            raise RuntimeError("index not loaded; call build_index or load_index")
        q = self.embed_text(query)
        D, I = self.index.search(q, k)
        res = []
        for d, i in zip(D[0], I[0]):
            if i < 0: continue
            res.append((self.paths[i], float(d)))
        return res
