import numpy as np
import copy

DEFAULT_CLIP_NORM = 0.5
DEFAULT_NOISE_MULTIPLIER = 0.05

def clip_and_add_noise(delta: dict, clip_norm=DEFAULT_CLIP_NORM, noise_multiplier=DEFAULT_NOISE_MULTIPLIER):
    keys = sorted(delta.keys())
    
    # Flatten
    flat_vectors = []
    shapes = {}
    for k in keys:
        arr = np.array(delta[k], dtype=np.float32)
        shapes[k] = arr.shape
        flat_vectors.append(arr.flatten())
    
    flat_delta = np.concatenate(flat_vectors)
    original_l2_norm = float(np.linalg.norm(flat_delta))
    
    # Clip
    clip_factor = min(1.0, clip_norm / (original_l2_norm + 1e-12))
    flat_delta = flat_delta * clip_factor
    
    # Noise
    noise_std = noise_multiplier * clip_norm
    noise = np.random.normal(loc=0.0, scale=noise_std, size=flat_delta.shape)
    flat_delta = flat_delta + noise
    
    post_noise_l2_norm = float(np.linalg.norm(flat_delta))
    
    # Unflatten
    noised_delta = {}
    offset = 0
    for k in keys:
        size = np.prod(shapes[k])
        noised_delta[k] = flat_delta[offset:offset+size].reshape(shapes[k]).tolist()
        offset += size
        
    stats = {
        "original_l2_norm": original_l2_norm,
        "clip_factor_applied": clip_factor,
        "noise_std_per_coordinate": noise_std,
        "post_noise_l2_norm": post_noise_l2_norm
    }
    
    return noised_delta, stats
