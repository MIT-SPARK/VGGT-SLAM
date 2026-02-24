import os
import re
import numpy as np
import matplotlib
import scipy
import time
from PIL import Image
from torchvision import transforms as TF
import torch


def slice_with_overlap(lst, n, k):
    if n <= 0 or k < 0:
        raise ValueError("n must be greater than 0 and k must be non-negative")
    result = []
    i = 0
    while i < len(lst):
        result.append(lst[i:i + n])
        i += max(1, n - k)  # Ensure progress even if k >= n
    return result


def sort_images_by_number(image_paths):
    def extract_number(path):
        filename = os.path.basename(path)
        # Look for digits followed immediately by a dot and the extension
        match = re.search(r'\d+(?:\.\d+)?(?=\.[^.]+$)', filename)
        return float(match.group()) if match else float('inf')

    return sorted(image_paths, key=extract_number)

def downsample_images(image_names, downsample_factor):
    """
    Downsamples a list of image names by keeping every `downsample_factor`-th image.
    
    Args:
        image_names (list of str): List of image filenames.
        downsample_factor (int): Factor to downsample the list. E.g., 2 keeps every other image.

    Returns:
        list of str: Downsampled list of image filenames.
    """
    return image_names[::downsample_factor]

def decompose_camera(P, no_inverse=False):
    """
    Decompose a 3x4 or 4x4 camera projection matrix P into intrinsics K,
    rotation R, and translation t.
    """
    if P.shape[0] != 3:
        P = P / P[-1,-1]
        P = P[0:3, :]

    # Ensure P is (3,4)
    assert P.shape == (3, 4)

    # Left 3x3 part
    M = P[:, :3]

    # RQ decomposition
    K, R = scipy.linalg.rq(M)

    # Make sure intrinsics have positive diagonal
    if K[0,0] < 0:
        K[:,0] *= -1
        R[0,:] *= -1
    if K[1,1] < 0:
        K[:,1] *= -1
        R[1,:] *= -1
    if K[2,2] < 0:
        K[:,2] *= -1
        R[2,:] *= -1

    scale = K[2,2]
    # print("Scale factor from K[2,2]:", scale)
    if not no_inverse:
        R = np.linalg.inv(R)
        t = -R @ np.linalg.inv(K) @ P[:, 3]
    else:
        t = np.linalg.inv(K) @ P[:, 3]
    K = K / scale
    
    return K, R, t, scale

def compute_image_embeddings(model, preprocess, image_paths, batch_size=64, device="cuda"):
    all_embs = []

    # Load all images into memory (PIL -> tensor)
    imgs = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        imgs.append(preprocess(img))

    # Stack into a single tensor
    imgs = torch.stack(imgs).to(device)

    # Loop over batches
    with torch.no_grad():
        for i in range(0, len(imgs), batch_size):
            batch = imgs[i : i + batch_size]
            emb = model.encode_image(batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embs.append(emb.cpu())

    # Combine into one (N, D) array
    return torch.cat(all_embs, dim=0).numpy()

def compute_text_embeddings(clip_model, tokenizer, text, device="cuda"):
    with torch.no_grad():
        text_tokens = tokenizer([text]).to(device)
        text_emb = clip_model.encode_text(text_tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        return text_emb.cpu().numpy()

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors a and b.
    """
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return a @ b.T

def normalize_to_sl4(H):
    """
    Normalize a 4x4 homography matrix H to be in SL(4).
    """
    det = np.linalg.det(H)
    if det == 0:
        raise ValueError("Homography matrix is singular and cannot be normalized.")
    scale = det ** (1/4)
    H_normalized = H / scale
    return H_normalized

def compute_obb_from_points(points: np.ndarray):
    """
    Compute an oriented bounding box (OBB) for a Nx3 point cloud.

    Returns:
        center      : (3,) world-space center of OBB
        extent      : (3,) lengths of OBB along its principal axes
        rotation    : (3,3) rotation matrix (columns = principal axes)
    """
    assert points.ndim == 2 and points.shape[1] == 3, "Input must be Nx3 points"

    # Remove NaN/inf if any
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise ValueError("Point cloud is empty or invalid")

    # 1. Compute centroid
    centroid = points.mean(axis=0)

    # 2. PCA on centered points
    centered = points - centroid
    cov = np.cov(centered, rowvar=False)

    # Eigen decomposition (sorted by eigenvalue descending)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    rotation = eigvecs  # columns = principal axes (R)

    # 3. Project points to PCA frame
    points_local = centered @ rotation

    # 4. Compute min/max in PCA frame → box extents
    min_corner = points_local.min(axis=0)
    max_corner = points_local.max(axis=0)
    extent = max_corner - min_corner

    # 5. Compute box center (local), then world
    center_local = 0.5 * (min_corner + max_corner)
    center_world = centroid + center_local @ rotation.T

    return center_world, extent, rotation

def overlay_masks(image, masks):
    image = image.convert("RGBA")

    # masks: (N, 1, H, W) or (N, H, W)
    masks = (255 * masks.cpu().numpy()).astype(np.uint8)

    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    for mask, color in zip(masks, colors):
        # Ensure mask is 2D
        mask = np.squeeze(mask)
        # Now mask is shape (H, W)

        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)

    return image

class Accumulator:
    def __init__(self):
        self.total_time = 0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.total_time += (time.perf_counter() - self.start)
