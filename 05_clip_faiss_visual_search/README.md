# Project 5 - Visual Search with CLIP Embeddings + FAISS

Build a multimodal image search: index a folder of images with CLIP, then search by natural-language text queries (e.g. `"red car"`, `"person with backpack"`) to retrieve the most semantically similar images using FAISS (inner product on normalized embeddings).

## Install

```bash
pip install torch open_clip_torch faiss-cpu opencv-python pillow
```

Note: `faiss-gpu` is faster if you have CUDA; `faiss-cpu` works fine for small sets.

## 1) Prepare images

Create an `images/` folder and drop some photos (JPG/PNG/etc).

## 2) Build index

```bash
python search.py --index --images images --prefix index
```

This creates:
- `index.faiss` - FAISS vector index (dimension 512 for ViT-B-32)
- `index_paths.pkl` - mapping of index rows to image file paths

## 3) Search

```bash
python search.py --query "red car" --k 5
python search.py --query "group of people walking" --k 5 --show
```

With `--show`, top results open in OpenCV windows (press any key to close).

## How it works

- **CLIP (Contrastive Language–Image Pretraining)** learns a shared embedding space for images and text. Images and text that are semantically related end up close to each other in that space.
- **Embeddings**: we encode every image with CLIP's image encoder and every query text with its text encoder. We L2-normalize so that cosine similarity becomes inner product.
- **FAISS**: `IndexFlatIP` does exact inner-product search. For larger collections (10k–1M) you can swap to IVF/PQ/HNSW for speed; FlatIP is perfect here and easy to understand.
- **Search**: given text query `q`, we get `q_e = encode_text(q)`, then FAISS returns `k` image indices `i` with highest `q_e · I_i` (closest in cosine space).

## Embeddings, Vector DBs & Multimodal - explained

- **Embeddings**: dense vectors that capture meaning. Similar concepts → nearby vectors. CLIP produces multimodal embeddings (same space for text/images).
- **Vector databases**: systems optimized for similarity search at scale (FAISS, Chroma, Weaviate, Qdrant, Milvus). They support ANN (Approximate Nearest Neighbor) to trade tiny recall for big speedups.
- **Multimodal models**: handle multiple modalities (text+image) in one model. CLIP is zero-shot - no task-specific training needed to search "red car" against arbitrary images.
- **Zero-shot retrieval**: you don't need labels; natural language is the query interface.

## Interview talking points

- **Why normalize + inner product?** Cosine similarity = (a·b)/(||a|| ||b||). With unit vectors, equals inner product. FAISS IndexFlatIP is efficient and correct here.
- **Exact vs ANN**: IndexFlatIP is exact. For production with millions of images, use IVF (inverted lists) + PQ (product quantization) or HNSW to reduce memory/time.
- **CLIP strengths**: open vocabulary, works without retraining. Weaknesses: fine-grained attributes (license plate numbers, exact text) can be fuzzy; resolution/occlusion affect results.
- **Use cases**: visual search, content moderation triage, duplicate detection (image→image), ad targeting, and "find similar" in DAMs.
- **Safety/ethics**: be mindful of copyrighted images; avoid embedding/searching private PII unless necessary.
