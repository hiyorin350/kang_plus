import numpy as np
from sub_conversions import *
import cv2

def clip_lab_within_rgb_gamut(lab_image, step=1):
    """
    Lab色空間の画像をRGB色空間に収まるように調整する。
    この際、L（明度）とh（色相）を保持し、C（彩度）のみを調整する。

    引数:
    lab_image (numpy.ndarray): Lab色を表す画像配列。
    step (int): Cを減少させる際のステップサイズ。

    戻り値:
    numpy.ndarray: 調整されたLab画像。
    """
    # 画像の高さ、幅、チャンネル数を取得
    height, width, channels = lab_image.shape

    # 出力用のLab画像を初期化
    clipped_lab_image = np.copy(lab_image)

    for i in range(height):
        for j in range(width):
            L, a, b = lab_image[i, j]

            # 初期のC（彩度）とh（色相）を計算
            C = np.sqrt(a**2 + b**2)
            h = np.degrees(np.arctan2(b, a))

            # LabからRGBへ変換
            lab_color_reshaped = np.array([L, a, b], dtype=np.float32).reshape(1, 1, 3)
            rgb_color = lab_to_rgb(lab_color_reshaped)

            # Cを減少させながらRGB色空間内に収まるかチェック
            while not np.all((rgb_color >= 0) & (rgb_color <= 1)):
                C -= step
                if C < 0:
                    C = 0
                    break

                a = C * np.cos(np.radians(h))
                b = C * np.sin(np.radians(h))
                lab_color_reshaped = np.array([L, a, b], dtype=np.float32).reshape(1, 1, 3)
                rgb_color = lab_to_rgb(lab_color_reshaped)

            # 調整されたLab値を出力画像に設定
            clipped_lab_image[i, j] = np.array([L, a, b], dtype=np.float32)

    return clipped_lab_image

def check_gamut(lab_image):
    """
    Lab色空間の画像内の色がRGB色空間内に収まるかをチェックし、
    収まらない場合は赤色に変換する。

    引数:
    lab_image (numpy.ndarray): Lab色を表す画像配列。

    戻り値:
    numpy.ndarray: 調整されたLab色の画像。
    """
    # 出力用のLab画像を初期化
    adjusted_lab_image = np.copy(lab_image)
    
    # RGBに変換
    rgb_image = lab_to_rgb(lab_image)
    
    # RGB範囲外の場合は赤色に変換
    out_of_gamut = ~((rgb_image >= 0) & (rgb_image <= 1.1)).all(axis=-1)
    
    # 範囲外の画素の元のRGB値と変換後のRGB値を記録
    out_of_gamut_indices = np.where(out_of_gamut)
    for idx in zip(*out_of_gamut_indices):
        original_rgb = rgb_image[idx]
        adjusted_lab_image[idx] = np.array([53, 80, 67], dtype=np.float32)  # 赤色のLab値
        adjusted_rgb = lab_to_rgb(adjusted_lab_image[idx].reshape(1, 1, 3)).flatten()
        print(f"飛び出した画素: 位置 {idx}, 元のRGB値 {original_rgb}, 変換後のRGB値 {adjusted_rgb}")
    
    return adjusted_lab_image

def rotate_hue_lab(lab_color, angle):
    """
    LAB色空間において、指定した角度だけ色相（h）を回転させる。

    引数:
    lab_color (numpy.ndarray): Lab色を表す配列。
    angle (float): 回転させる角度（度単位）。

    戻り値:
    numpy.ndarray: 色相が回転されたLab色。
    """
    l = lab_color[:, 0]
    a = lab_color[:, 1]
    b = lab_color[:, 2]

    h = np.degrees(np.arctan2(b, a))  # 現在の色相を計算
    h_rotated = (h + angle) % 360  # 角度を加え、360度で割った余りを取る

    # 回転した色相で新しいaとbを計算
    C = np.sqrt(a**2 + b**2)
    a_rotated = C * np.cos(np.radians(h_rotated))
    b_rotated = C * np.sin(np.radians(h_rotated))

    return np.stack([l, a_rotated, b_rotated], axis=-1)

if __name__ == "__main__":
    # 画像の読み込み
    image = cv2.imread('/Users/hiyori/kang_plus_lab/images/map_pjt.ppm')

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width, channels = image.shape
    N = height * width

    lab_image = rgb_to_lab(rgb_image)

    flat_lab_image = lab_image.reshape(N, 3)

    rotated_lab = rotate_hue_lab(flat_lab_image, 180)

    cliped_lab = clip_lab_within_rgb_gamut(rotated_lab.reshape(height, width, 3))

    check_gamut(rotated_lab.reshape(height, width, 3))

    rgb_out = lab_to_rgb(cliped_lab)
    rgb_out = (rgb_out * 255).astype(np.uint8)

    img_out = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)

    # 回転された画像を表示
    cv2.imwrite('/Users/hiyori/kang_plus_lab/images/map_pjt_rev.ppm', img_out)
    cv2.imshow('map', img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
