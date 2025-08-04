import os
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from mmseg.apis import init_segmentor, inference_segmentor

# === Color Palette for Binary (0 = background, 1 = cell) ===
BINARY_PALETTE = [
    [0, 0, 0],        # 0: Background (black)
    [255, 255, 255]   # 1: Cell (white)
]

# === Merge all foreground classes (1-7) to 1 ===
def convert_multiclass_to_binary(mask):
    return np.where(mask > 0, 1, 0).astype(np.uint8)

# === Color segmentation mask ===
def color_segmentation(mask, palette):
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_idx in range(len(palette)):
        colored_mask[mask == class_idx] = palette[class_idx]
    return colored_mask

# === Overlay image with mask ===
def overlay_image(image, mask, alpha=0.5):
    image = np.array(image)
    colored_mask = color_segmentation(mask, BINARY_PALETTE)
    return Image.fromarray(cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0))

# === Concatenate images side by side ===
def concat_images(images, axis=1):
    widths, heights = zip(*(i.size for i in images))
    if axis == 1:
        new_img = Image.new('RGB', (sum(widths), max(heights)))
        x_offset = 0
        for img in images:
            new_img.paste(img, (x_offset, 0))
            x_offset += img.size[0]
    else:
        new_img = Image.new('RGB', (max(widths), sum(heights)))
        y_offset = 0
        for img in images:
            new_img.paste(img, (0, y_offset))
            y_offset += img.size[1]
    return new_img

# === Config and checkpoint ===
config_file = 'local_configs/segformer/B1/segformer.b1.512x512.hemato.160k.py'
checkpoint_file = 'work_dirs/segformer_b1_hemato/latest.pth'

# === Input and output directories ===
img_dir = '/media/iml/cv-lab/Datasets_B_cells/Hemato_Data/inference'
out_dir = 'inference_results_binary'
os.makedirs(out_dir, exist_ok=True)

# === Initialize model ===
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# === Inference loop ===
for img_name in sorted(os.listdir(img_dir)):
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(img_dir, img_name)
    result = inference_segmentor(model, img_path)
    mask = result[0].astype(np.uint8)

    # Convert multiclass to binary
    binary_mask = convert_multiclass_to_binary(mask)

    # Load and process images
    original_img = Image.open(img_path).convert('RGB')
    colored_mask = Image.fromarray(color_segmentation(binary_mask, BINARY_PALETTE))
    overlayed_img = overlay_image(original_img, binary_mask, alpha=0.4)

    # Create and save combined result
    comparison = concat_images([original_img, colored_mask, overlayed_img])
    base_name = os.path.splitext(img_name)[0]
    comparison.save(os.path.join(out_dir, f'{base_name}_comparison.png'))
    colored_mask.save(os.path.join(out_dir, f'{base_name}_mask.png'))
    overlayed_img.save(os.path.join(out_dir, f'{base_name}_overlay.png'))

    print(f"✅ Processed: {img_name}")

print("\n✅ Inference completed!")
