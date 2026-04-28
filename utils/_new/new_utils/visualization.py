
import os
import torch
import torchvision
import cv2

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from skimage.segmentation import mark_boundaries

from .preprocessing import apply_universal_mask

def debug_save_pixels_interval(pixels_tensor, orig_tensor, output_dir, t_dim_index, filename="debug_blend_alpha_0.5.png"):
    """
    Extracts the VLM's pixel input tensor, un-normalizes it using the original 
    video's min/max to preserve the alpha-blend fading effect, and saves it.
    """
    pt = pixels_tensor.detach().cpu().float()
    pt_orig = orig_tensor.detach().cpu().float()
    # Realign dimensions to (Time, Channels, Height, Width)
    if t_dim_index == 1:
        frames = pt[0] 
        orig_frames = pt_orig[0]
    else:
        frames = pt[0].permute(1, 0, 2, 3) 
        orig_frames = pt_orig[0].permute(1, 0, 2, 3)
    
    #-- Calculate min/max from the original frames
    # --> Locks the "exposure" so faded/blurred frames actually look faded.
    amin = orig_frames.amin(dim=(0, 1, 2, 3), keepdim=True) # original absolute minimum tensor value
    amax = orig_frames.amax(dim=(0, 1, 2, 3), keepdim=True) # original absolute maximum tensor value
    frames -= amin
    frames /= (amax - amin) + 1e-5
    frames = torch.clip(frames, 0, 1) # Prevent out-of-bounds from baseline weirdness
    # Stitch frames horizontally into a single image
    grid = torchvision.utils.make_grid(frames, nrow=frames.shape[0], padding=2)
    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    torchvision.utils.save_image(grid, save_path)

def visualize_spix(video_array, baseline_array, tubelets, selected_tubes, output_path):
    masked_frames = cv_utils.apply_universal_mask(video_array, baseline_array, tubelets, selected_tubes)
    masked_frames[0].save(
        output_path, save_all=True, append_images=masked_frames[1:], duration=250, loop=0
    )

def visualize_frames(frames, output_path):
    frames[0].save(
        output_path, save_all=True, append_images=frames[1:], duration=250, loop=0
    )

def visualize_tubelets(video_array, tubelet_labels, output_path):
    visualized_video = []
    for i in range(len(video_array)):
        frame = video_array[i]
        label_slice = tubelet_labels[i]
        boundary_img = mark_boundaries(frame, label_slice)
        boundary_img_uint8 = (boundary_img * 255).astype(np.uint8)
        visualized_video.append(Image.fromarray(boundary_img_uint8))
    visualized_video[0].save(
        output_path, save_all=True, append_images=visualized_video[1:], duration=250, loop=0 
    )
    print(f"Saved visualization to {output_path}")


def visualize_heatmap(video_array, tubelet_labels, tubelet_scores, output_path, alpha=0.7, blur_fraction=0.15, gamma=1.0):
    visualized_video = []
    max_id = tubelet_labels.max()
    score_map = np.zeros(max_id + 1, dtype=np.float32)
    
    # Clip negative scores to 0 (we only care about positive signal)
    for tid, score in tubelet_scores.items():
        score_map[tid] = max(0.0, score)
        
    # Normalize the scores so the maximum value is exactly 1.0
    mask_max = score_map.max()
    if mask_max > 0:
        score_map = score_map / mask_max
        
    # Optional: Apply gamma to boost mid-tones if needed
    #score_map = np.power(score_map, gamma)
    heatmap_mask = score_map[tubelet_labels]
    H, W = video_array[0].shape[:2]
    k_size = int(min(H, W) * blur_fraction)
    k_size = k_size + 1 if k_size % 2 == 0 else k_size 
    k_size = max(15, k_size) 

    for i in range(len(video_array)):
        image = video_array[i].astype(np.float32) # Ensure float for smooth blending
        mask_frame = heatmap_mask[i]
        # Blur the 0-1 mask
        blurred_mask = cv2.GaussianBlur(mask_frame, (k_size, k_size), 0)
        # Convert to 0-255 for the colormap
        heatmap_uint8 = np.uint8(255 * blurred_mask)
        heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32)
        # We scale the alpha by the blurred mask itself.
        # Where the mask is 0, dynamic_alpha is 0 (100% original image).
        # Where the mask is 1, dynamic_alpha is `alpha` (e.g., 60% heatmap, 40% image).
        dynamic_alpha = blurred_mask[..., np.newaxis] * alpha
        overlay = (1.0 - dynamic_alpha) * image + dynamic_alpha * heatmap
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        visualized_video.append(Image.fromarray(overlay))
        
    visualized_video[0].save(
        output_path, save_all=True, append_images=visualized_video[1:], duration=250, loop=0
    )
    print(f"Saved heatmap visualization to {output_path}")




def visualize_gradients(gradients, frames, tubelets, save_folder, step, title="IG_Step", alpha=0.6, blur_fraction=0.08):
    """
    Overlays tubelet gradients onto original frames using cv2/PIL for fast GIF generation.
    Blue = Negative (suppresses output), Red = Positive (supports output).
    Unimportant areas (zero gradient) remain the original grayscale frame.
    """
    #-- Map 1D tubelet gradients back to the 3D video space (T, H, W)
    if torch.is_tensor(gradients):
        gradients = gradients.detach().cpu().numpy()
    if torch.is_tensor(tubelets):
        tubelets = tubelets.detach().cpu().numpy()
        
    spatial_grads = gradients[tubelets].astype(np.float32)
    T, H, W = spatial_grads.shape
    
    #-- Determine symmetric limits for the diverging colormap
    vmax = np.max(np.abs(spatial_grads))
    if vmax == 0: vmax = 1e-9 # Prevent division by zero
    
    #-- Dynamic blur kernel size based on frame resolution
    k_size = int(min(H, W) * blur_fraction)
    k_size = k_size + 1 if k_size % 2 == 0 else k_size 
    k_size = max(5, k_size) 

    visualized_video = []
    cmap = plt.get_cmap('coolwarm') # Still the easiest way to get Blue-White-Red

    for t in range(T):
        #-- Extract and format the base image
        image = frames[t]
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
            if image.ndim == 3 and image.shape[0] == 3: # If channels first (C, H, W)
                image = np.transpose(image, (1, 2, 0))
        else:
            # ---> FIX: Convert PIL Image (or other types) to NumPy array <---
            image = np.array(image)
                
        # Ensure image is in 0-255 float range for blending
        if image.max() <= 1.0:
            image = image * 255.0
        image = image.astype(np.float32)
        
        #-- Convert base image to 3-channel grayscale so the Red/Blue heatmap pops
        # Check if grayscale already (shape (H, W)) to prevent index errors
        if image.ndim == 2:
            image = image[..., np.newaxis]
            
        gray_image = np.mean(image, axis=-1, keepdims=True)
        gray_image = np.repeat(gray_image, 3, axis=-1)
        
        #-- Process the gradient mask
        grad_frame = spatial_grads[t]
        
        # Blur the raw gradients to smooth the blocky tubelets
        blurred_grad = cv2.GaussianBlur(grad_frame, (k_size, k_size), 0)
        
        # Calculate absolute magnitude for the alpha mask (0.0 to 1.0)
        magnitude = np.abs(blurred_grad) / vmax
        magnitude = np.clip(magnitude, 0, 1)
        
        # Normalize gradients to [0, 1] for the colormap (-vmax -> 0.0, 0 -> 0.5, +vmax -> 1.0)
        norm_grad = (blurred_grad / vmax + 1.0) / 2.0
        norm_grad = np.clip(norm_grad, 0, 1)
        
        # Apply colormap to get RGB overlay (drop the alpha channel from cmap)
        heatmap_rgba = cmap(norm_grad)
        heatmap_rgb = (heatmap_rgba[..., :3] * 255.0).astype(np.float32)
        
        #-- Perform dynamic alpha blending
        # Where magnitude is 0, alpha is 0 (100% original frame)
        # Where magnitude is 1, alpha is `alpha` (e.g., 60% heatmap, 40% frame)
        dynamic_alpha = magnitude[..., np.newaxis] * alpha
        overlay = (1.0 - dynamic_alpha) * gray_image + dynamic_alpha * heatmap_rgb
        
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        visualized_video.append(Image.fromarray(overlay))

    # --- SAVE AS GIF ---
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"{title}_step_{step:03d}.gif")
    
    visualized_video[0].save(
        save_path, 
        save_all=True, 
        append_images=visualized_video[1:], 
        duration=250, # 4 FPS 
        loop=0
    )


def visualize_interaction_matrix(interaction_matrix, output_path):
    plt.figure(figsize=(10, 8))
    max_val = np.max(np.abs(interaction_matrix))
    if max_val == 0:
        max_val = 1.0 
    sns.heatmap(
        interaction_matrix, cmap="RdYlGn", center=0, vmin=-max_val, vmax=max_val,
        annot=True, fmt=".3f", linewidths=.5,
        cbar_kws={'label': 'Interaction Index (Red=Redundant (lim -2), Green=Synergistic (lim 2))'}
    )
    plt.title("Frame Pairwise Interactions (Shapley)")
    plt.xlabel("Frame Index J")
    plt.ylabel("Frame Index I")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

