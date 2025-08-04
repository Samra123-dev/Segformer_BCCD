from mmseg.apis import init_segmentor, inference_segmentor
import os
import numpy as np
from PIL import Image
from scipy.ndimage import label

# === Generate random colors ===
def random_color_map(n):
    np.random.seed(42)
    return [tuple(np.random.randint(50, 256, size=3)) for _ in range(n)]

# === Color each connected cell, black background ===
def color_cells_with_black_background(mask, alpha=1.0):
    labeled_mask, num_cells = label(mask == 1)

    # Start with black background
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Assign random color to each cell
    colors = random_color_map(num_cells + 1)

    for i in range(1, num_cells + 1):  # skip background label 0
        cell_mask = labeled_mask == i
        for c in range(3):
            colored_mask[:, :, c][cell_mask] = int(colors[i][c] * alpha)

    return colored_mask

# === Concatenate side-by-side ===
def concat_side_by_side(img1, img2):
    arr1 = np.array(img1)
    arr2 = np.array(img2)

    h1, w1, _ = arr1.shape
    h2, w2, _ = arr2.shape
    max_height = max(h1, h2)

    if h1 != max_height:
        scale = max_height / h1
        arr1 = np.array(Image.fromarray(arr1).resize((int(w1 * scale), max_height)))
    if h2 != max_height:
        scale = max_height / h2
        arr2 = np.array(Image.fromarray(arr2).resize((int(w2 * scale), max_height)))

    combined = np.concatenate((arr1, arr2), axis=1)
    return Image.fromarray(combined)

# === Config and checkpoint ===
config_file = '/media/iml/cv-lab/Segformer/SegFormer/local_configs/segformer/B1/segformer.b1.512x512.bccd.160k.py'
checkpoint_file = '/media/iml/cv-lab/Segformer/SegFormer/work_dirs/segformer.b1.512x512.bccd.160k/latest.pth'

# === Input and output directories ===
img_dir = '/media/iml/cv-lab/Datasets_B_cells/BCCD/inference'
out_dir = 'inference_results'
os.makedirs(out_dir, exist_ok=True)

# === Init model ===
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# === Inference loop ===
for img_name in os.listdir(img_dir):
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(img_dir, img_name)
    result = inference_segmentor(model, img_path)

    original_img = Image.open(img_path).convert('RGB')
    mask = result[0].astype(np.uint8)

    # Apply connected component coloring
    colored_seg = color_cells_with_black_background(mask, alpha=1.0)
    colored_seg_img = Image.fromarray(colored_seg)

    # Side-by-side comparison
    combined_img = concat_side_by_side(original_img, colored_seg_img)

    # Save result
    base_name = os.path.splitext(img_name)[0]
    out_path = os.path.join(out_dir, f'{base_name}_black_bg_colored_cells.png')
    combined_img.save(out_path)

    print(f"✅ Saved: {out_path}")
