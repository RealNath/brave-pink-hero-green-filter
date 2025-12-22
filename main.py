import sys
import numpy as np
from PIL import Image

# Hero Green (#1B602F, rgb(27, 96, 47))
SHADOW_COLOR = np.array([27, 96, 47])
# Brave Pink (#F784C5, rgb(247, 132, 197))
HIGHLIGHT_COLOR = np.array([247, 132, 197])

input_path = sys.argv[1]
output_path = sys.argv[2]

# 1. Load gambar
img = Image.open(input_path).convert('RGB')
# Ubah ke 0 (x1) - 1 (x2)
img_array = np.array(img) / 255.0

# 2. Konversi ke grayscale (Rec. 709)
luma_weights = np.array([0.2126, 0.7152, 0.0722])
luminance = img_array.dot(luma_weights)
# print(f"Luminance shape: {luminance.shape}")

# 3. Auto-Contrast
min_lum = np.min(luminance)
max_lum = np.max(luminance)
if min_lum != max_lum:
    luminance = (luminance - min_lum) / (max_lum - min_lum)
luminance = np.clip(luminance, 0, 1)

# 4. Duotone Mapping
# Ubah dimensi (H, W) jadi (H, W, 1)
t = luminance[:, :, np.newaxis]
# print(f"t shape: {t.shape}")

# Interpolasi linear
#  y    =      y1      + (x −x1) * ( (        y2      −      y1     ) / (x2−x1) )
duotone = SHADOW_COLOR + (t - 0) * ( (HIGHLIGHT_COLOR - SHADOW_COLOR) / (1 - 0) )
# print(f"Duotone shape: {duotone.shape}")

# 5. Save
result = Image.fromarray(duotone.astype(np.uint8))
result.save(output_path)
print(f"Gambar berhasil diproses: '{input_path}' -> '{output_path}'")