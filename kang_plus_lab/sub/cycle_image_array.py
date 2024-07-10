import os
import sys
import numpy as np
import cv2
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import conversions
from cycle_hue import clip_lab_within_rgb_gamut

def lab_to_lch(lab_image):
    """
    Lab画像をLCH画像に変換する関数。
    """
    # Labチャンネルを分割
    L, a, b = cv2.split(lab_image)
    
    # Lはそのまま
    L = L.astype(np.float32)
    
    # C（彩度）を計算
    C = np.sqrt(a**2 + b**2)
    
    # H（色相）を計算
    H = np.arctan2(b, a)
    H = np.degrees(H)
    H = np.where(H < 0, H + 360, H)
    
    # デバッグ用: 型を一致させる
    C = C.astype(L.dtype)
    H = H.astype(L.dtype)

    return cv2.merge([L, C, H])

def lch_to_lab(lch_image):
    """
    LCH画像をLab画像に変換する関数。
    """
    L, C, H = cv2.split(lch_image)
    H = np.radians(H)
    a = C * np.cos(H)
    b = C * np.sin(H)
    return cv2.merge([L, a, b])

# 画像の読み込み
image = cv2.cvtColor(cv2.imread('images/chart26/chart26.ppm'), cv2.COLOR_BGR2RGB)
height, width, channels = image.shape

# RGBからLabへの変換
lab_image = conversions.rgb_to_lab(image)
lch_image = lab_to_lch(lab_image)

# 色相を15度ずつ変えて画像を生成
for i in range(0, 360, 15):
    adjusted_lch_image = lch_image.copy()
    adjusted_lch_image[:, :, 2] = (adjusted_lch_image[:, :, 2] + i) % 360
    
    # Labに変換してクリップ処理
    adjusted_lab_image = lch_to_lab(adjusted_lch_image)
    clipped_lab_image = clip_lab_within_rgb_gamut(adjusted_lab_image)
    
    # クリップ後のLab画像をRGBに変換
    img_out_rgb = conversions.lab_to_rgb(clipped_lab_image)
    img_out = cv2.cvtColor((img_out_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    output_filename = f'images/chart26/rotates/chart26_kang_plus_rotate_lab_{i}.ppm'
    # cv2.imwrite(output_filename, img_out)
    print(f'Saved: {output_filename}')

print("All images have been saved.")