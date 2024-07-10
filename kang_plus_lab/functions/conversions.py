import numpy as np
import cv2
def rgb_to_lab(rgb_image):
    """
    rgb_image:np.uint8型
    """
    assert rgb_image.dtype == np.uint8, "画像はnp.uint8型である必要があります"
    
    # 正規化とリニア化
    rgb_image = rgb_image.astype(np.float32) / 255.0
    rgb_linear = np.where(rgb_image > 0.04045, ((rgb_image + 0.055) / 1.055) ** 2.4, rgb_image / 12.92)
    
    # RGBからXYZへの変換
    mat_rgb_to_xyz = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    xyz = np.dot(rgb_linear, mat_rgb_to_xyz.T)
    
    # XYZからL*a*b*への変換
    xyz /= np.array([0.95047, 1.00000, 1.08883])
    xyz = np.where(xyz > (6/29) ** 3, xyz ** (1/3), (xyz * (29/6) ** 2 + 16/116) * 3/29)
    L = 116 * xyz[:, :, 1] - 16
    a = 500 * (xyz[:, :, 0] - xyz[:, :, 1])
    b = 200 * (xyz[:, :, 1] - xyz[:, :, 2])
    
    lab_image = np.stack([L, a, b], axis=-1)
    return lab_image

def lab_to_rgb(lab):
    """
    L*a*b*からRGB色空間への逆変換を行う。
    labはL, a, bの値を持つ3要素のリストまたはNumPy配列。
    """
    # L*a*b*からXYZへの変換
    def lab_to_xyz(l, a, b):
        y = (l + 16) / 116
        x = a / 500 + y
        z = y - b / 200

        xyz = np.array([x, y, z])
        mask = xyz > 6/29
        xyz[mask] = xyz[mask] ** 3
        xyz[~mask] = (xyz[~mask] - 16/116) / 7.787

        # D65光源の参照白
        xyz_ref_white = np.array([0.95047, 1.00000, 1.08883])
        xyz = xyz * xyz_ref_white
        return xyz

    xyz = lab_to_xyz(*lab)

    # XYZからリニアRGBへの変換
    mat_xyz_to_rgb = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ])
    rgb_linear = np.dot(mat_xyz_to_rgb, xyz)

    # リニアRGBからsRGBへのガンマ補正
    def gamma_correction(channel):
        return np.where(channel > 0.0031308, 1.055 * (channel ** (1/2.4)) - 0.055, 12.92 * channel)

    rgb = gamma_correction(rgb_linear)

    # RGB値を[0, 255]の範囲にクリッピングして整数に変換
    rgb_clipped = np.clip(rgb * 255, 0, 255).astype(np.uint8)

    return rgb_clipped

def lab_to_lch(lab_image):
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
    
    return cv2.merge([L, C, H])

def lch_to_lab(lch_image):
    # LChチャンネルを分割
    L, C, H = cv2.split(lch_image)
    
    # Lはそのまま
    L = L.astype(np.float32)
    
    # Hをラジアンに変換
    H = np.radians(H)
    
    # aとbを計算
    a = C * np.cos(H)
    b = C * np.sin(H)
    
    return cv2.merge([L, a, b])
