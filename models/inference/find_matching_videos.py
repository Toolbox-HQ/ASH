import os
os.environ["OPENBLAS_NUM_THREADS"] = "64"
import torch
from transformers import AutoImageProcessor, AutoModel
import json
import glob
from torchcodec.decoders import VideoDecoder
import torch.nn.functional as F
from tqdm import tqdm
from models.util.misc import local_model_map
import torch.distributed as dist

def get_embeddings(model, processor, path: str):
    decoder = VideoDecoder(path)
    duration = float(decoder.metadata.duration_seconds)
    t, timestamps = 0, []
    while t < duration:
        timestamps.append(t)
        t += 2
    frames = decoder.get_frames_played_at(seconds=timestamps).data

    inputs = processor(images=frames, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = outputs.last_hidden_state[:, 0]  # CLS token
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    return image_embeds.cpu(), frames

def dino_embeddings_every(video_path: str, model_id: str = "ASH/assets/dinov2-base", device=None):
    model_id = local_model_map(model_id)
    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
    model = AutoModel.from_pretrained(model_id).eval()
    if device is not None:
        model = model.to(device)
    embeds, frames = get_embeddings(model, processor, video_path)
    return embeds, frames

def load_embeddings_and_metadata(folder_path: str):
    embed_files = sorted(glob.glob(os.path.join(folder_path, "*.pt")))
    all_meta, tensors = [], []

    bad_count = 0
    for ef in tqdm(embed_files):
        base = os.path.basename(ef).rsplit(".pt", 1)[0]
        mf = os.path.join(folder_path, f"{base}.json")
        if not os.path.exists(mf):
            continue
        emb = torch.load(ef, map_location="cpu")
        if not isinstance(emb, torch.Tensor):
            emb = torch.as_tensor(emb)
        try:
            with open(mf, "r") as f:
                meta = json.load(f)
            tensors.append(emb.float())
            all_meta.extend(meta)
        except:
            #raise ValueError(f'Error: {mf} did not load correctly')
            bad_count += 1
        
    print(f'BAD: {bad_count} -- investigate')
    
    out_tensor = F.normalize(torch.cat(tensors, dim=0), p=2, dim=1, eps=1e-12)
    return all_meta, out_tensor

def _segments_by_video(meta):
    segs = []
    if not meta:
        return segs
    start = 0
    cur = meta[0]["video_path"]
    for i in range(1, len(meta)):
        if meta[i]["video_path"] != cur:
            segs.append((start, i - 1))
            start = i
            cur = meta[i]["video_path"]
    segs.append((start, len(meta) - 1))
    return segs

NEG_INF = float("-inf")

# Cap on V * e_max * |Q| cells per batch, which bounds the three large GPU
# allocations in _greedy_window_scores (the similarity block, the mask, and the
# per-step masked copy). ~200M cells is ~2.5GB peak at float32.
CELL_BUDGET = 200_000_000


def _greedy_window_scores(sim_block: torch.Tensor, w_r: int) -> torch.Tensor:
    """Algorithm 3, lines 9-19, batched over videos and over window positions.

    sim_block: [V, e_max, Q], where sim_block[v, c, r] is the cosine similarity
    between database row c of video v and query frame r. Rows past a video's
    true length must already be filled with -inf, which drives every window
    overlapping the padding to -inf so it loses the max on line 18.

    Returns z* per video, shape [V]. A video shorter than w_r scores -inf.

    Each window position holds its own `used` mask, so the row suppression on
    line 16 (s_{r*,:} <- -inf) is scoped to one window and never leaks into the
    next -- the reference recomputes the matching from a fresh slice per window.
    Step j of every window is evaluated at once: window t looks at database
    column t + j, so the slice sim_block[:, j:j+nwin, :] is exactly the j-th
    column of every window, stacked.
    """
    V, e_max, Q = sim_block.shape
    nwin = e_max - w_r + 1
    if nwin <= 0:
        return torch.full((V,), NEG_INF, device=sim_block.device)

    used = torch.zeros((V, nwin, Q), dtype=torch.bool, device=sim_block.device)
    z = torch.zeros((V, nwin), dtype=sim_block.dtype, device=sim_block.device)
    for j in range(w_r):
        col = sim_block[:, j:j + nwin, :].masked_fill(used, NEG_INF)  # line 14 candidates
        best, r_star = col.max(dim=2)                                 # line 14
        z += best                                                     # line 15
        used.scatter_(2, r_star.unsqueeze(2), True)                   # line 16
    return z.amax(dim=1)                                              # line 18


def _iter_corpus_videos(emb_dir: str):
    """Yield (video_path, embeddings [n, D], meta rows) for each complete video.

    Shards are streamed one at a time, and videos that straddle a shard boundary
    are stitched back together -- in this corpus roughly one video per shard
    boundary is split, and scoring the halves separately would truncate its
    windows.
    """
    num_files = sum(1 for entry in os.scandir(emb_dir) if entry.is_file()) // 2
    pending_path, pending_emb, pending_meta = None, [], []

    def _joined(pieces):
        # Only a straddling video has more than one piece; concatenating in the
        # common single-piece case would copy the whole corpus for nothing.
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)

    for i in range(num_files):
        meta, emb = load_embedding_metadata_pair(f"{emb_dir}/{i}")
        print(f"[GPU {dist.get_rank()} RETRIEVAL] Loaded embeddings {i} from disk")
        emb = emb.float()
        segments = _segments_by_video(meta)
        for k, (s, e) in enumerate(segments):
            path = meta[s]["video_path"]
            if pending_path is not None and path != pending_path:
                yield pending_path, _joined(pending_emb), pending_meta
                pending_path, pending_emb, pending_meta = None, [], []
            pending_path = path
            rows = emb[s:e + 1]
            if k == len(segments) - 1:
                # only this one outlives the shard tensor, so only it needs a copy
                rows = rows.clone()
            pending_emb.append(rows)
            pending_meta.extend(meta[s:e + 1])
        del emb

    if pending_path is not None:
        yield pending_path, _joined(pending_emb), pending_meta


def _video_length_seconds(meta_rows) -> float:
    fps = float(meta_rows[0]["video_fps"])
    return float(meta_rows[-1]["sampled_frame_index"]) / fps


def _score_batch(query_embed: torch.Tensor, batch, w_r: int, device):
    """Score one length-bucketed batch of videos. batch: [(path, emb, length_s)]."""
    e_max = max(emb.shape[0] for _, emb, _ in batch)
    V, D = len(batch), query_embed.shape[1]

    padded = torch.zeros((V, e_max, D), dtype=torch.float32, device=device)
    for v, (_, emb, _) in enumerate(batch):
        padded[v, :emb.shape[0]].copy_(emb)

    # line 8: S <- Q E^T, held transposed as [V, e_max, Q] so that step j of
    # every window is a contiguous slice along the database axis.
    sim_block = padded @ query_embed.t()
    del padded
    for v, (_, emb, _) in enumerate(batch):
        sim_block[v, emb.shape[0]:] = NEG_INF

    scores = _greedy_window_scores(sim_block, w_r)
    del sim_block
    return [(path, float(score), length_s)
            for (path, _, length_s), score in zip(batch, scores.tolist())]


def greedy_retrieval_scores(query_embed: torch.Tensor, emb_dir: str, w_r: int,
                            device=None, buffer_rows: int = 400_000):
    """Algorithm 3, lines 5-21: score every corpus video by its best window.

    Returns [(video_path, z*, video_length_seconds)] for videos long enough to
    hold a window; shorter ones score -inf and are dropped.

    Videos are buffered and sorted by length before batching because corpus
    lengths are heavily skewed (p50 ~580 rows, max ~20k): batching in corpus
    order would pad most videos out to the longest one in the batch.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    query_embed = query_embed.float().to(device)
    n_query = query_embed.shape[0]

    results = []
    buffer, buffered_rows = [], 0

    def flush():
        nonlocal buffer, buffered_rows
        if not buffer:
            return
        buffer.sort(key=lambda item: item[1].shape[0])
        start = 0
        while start < len(buffer):
            end, e_max = start, 0
            while end < len(buffer):
                cand_max = max(e_max, buffer[end][1].shape[0])
                if end > start and (end - start + 1) * cand_max * n_query > CELL_BUDGET:
                    break
                e_max = cand_max
                end += 1
            results.extend(_score_batch(query_embed, buffer[start:end], w_r, device))
            start = end
        buffer, buffered_rows = [], 0

    for path, emb, meta_rows in _iter_corpus_videos(emb_dir):
        if emb.shape[0] < w_r:
            continue  # cannot hold a window; line 10 ranges over nothing
        buffer.append((path, emb, _video_length_seconds(meta_rows)))
        buffered_rows += emb.shape[0]
        if buffered_rows >= buffer_rows:
            flush()
    flush()

    return results


def load_embedding_metadata_pair(prefix):
    meta_path = f"{prefix}.json"
    tensor_path = f"{prefix}.pt"
    with open(meta_path, "r") as f:
        all_meta = json.load(f)
    embeddings = torch.load(tensor_path)
    return all_meta, embeddings

def get_videos(query_path: str, emb_dir: str, interval_length: int, num_intervals: int, max_vid_len: float = None):
    print(f"[GPU {dist.get_rank()} RETRIEVAL] Begin retrieval process")
    num_embeds_per_sample = interval_length // 2

    query_emb, frames = dino_embeddings_every(query_path)
    print(f"[GPU {dist.get_rank()} RETRIEVAL] Created dino embeddings")

    self_sim_matrix = (query_emb @ query_emb.T)
    self_similarity = self_sim_matrix.mean().item()
    print(f"[GPU {dist.get_rank()} RETRIEVAL] had self-similarity of {self_similarity}")
    gather_list = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gather_list, self_similarity)
    world_idx = torch.argmin(torch.tensor(gather_list, dtype=torch.float32)).item()
    print(f"[GPU {dist.get_rank()} RETRIEVAL] GPU {world_idx} has the lowest self-similarity from {gather_list} on {self_sim_matrix.size()} matrix")
    
    scored = greedy_retrieval_scores(query_emb, emb_dir, num_embeds_per_sample)
    print(f"[GPU {dist.get_rank()} RETRIEVAL] Scored {len(scored)} videos")

    scored.sort(key=lambda item: item[1], reverse=True)  # line 22: top-k(scores)
    print(f"[GPU {dist.get_rank()} RETRIEVAL] Completed similarity search")

    total_seconds = 0
    videos = []
    for video_path, score, video_length in scored:
        if len(videos) >= num_intervals:
            break
        if max_vid_len and video_length > max_vid_len:
            continue
        videos.append({
                "video_path": video_path,
                "score": score
            })
        total_seconds += video_length

    print(f"[GPU {dist.get_rank()} RETRIEVAL] Hrs: {total_seconds / 3600}")

    return videos, world_idx, query_emb, frames
