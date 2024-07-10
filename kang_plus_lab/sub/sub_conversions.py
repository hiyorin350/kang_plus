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

def lab_to_rgb(lab_image):
    """
    rgbは[0,1]で返却されます。
    """
    # L*a*b*からXYZへの変換
    y = (lab_image[:, :, 0] + 16) / 116
    x = lab_image[:, :, 1] / 500 + y
    z = y - lab_image[:, :, 2] / 200
    xyz = np.stack([x, y, z], axis=-1)
    
    xyz = np.where(xyz > 6/29, xyz ** 3, 3 * (6/29) ** 2 * (xyz - 4/29))
    xyz *= np.array([0.95047, 1.00000, 1.08883])
    
    # XYZからリニアRGBへの変換
    mat_xyz_to_rgb = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ])
    rgb_linear = np.dot(xyz, mat_xyz_to_rgb.T)

    # print(rgb_linear)

    # NaNをゼロに置き換え
    rgb_linear = np.nan_to_num(rgb_linear)

    # 負の値をゼロに置き換え
    rgb_linear = np.where(rgb_linear < 0, 0, rgb_linear)
    
    # リニアRGBからsRGBへのガンマ補正
    rgb = np.where(rgb_linear > 0.0031308, 1.055 * (rgb_linear ** (1/2.4)) - 0.055, 12.92 * rgb_linear)
    
    return rgb