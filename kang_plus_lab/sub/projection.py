import numpy as np
import cv2
from sub_conversions import *

def angle_to_normal_vector(angle):
    """
    2次元空間において、指定された角度での直線の法線ベクトルを計算する。
    
    :param angle: 直線がX軸と成す角度（度単位）
    :return: 法線ベクトル（numpy配列）
    """
    # 角度をラジアンに変換
    angle_rad = np.radians(angle)
    
    # 直線の法線ベクトルの計算
    # 直線が成す角度に90度を加える（垂直な方向）
    nx = np.cos(angle_rad + np.pi / 2)
    ny = np.sin(angle_rad + np.pi / 2)
    
    return np.array([0, nx, ny])

def project_pixels_to_color_plane(image, u):
    """
    射影された画像を返す関数。

    :param image: 入力画像（CIE L*a*b* 色空間）
    :param u: 色平面の法線ベクトル
    :return: 射影された画像
    """
    # 画像の形状を取得
    height, width, _ = image.shape

    # 射影された画像を格納するための配列を初期化
    projected_image = np.zeros_like(image)

    # 各画素に対して射影を行う
    for i in range(height):
        for j in range(width):
            # 画素の色ベクトルを取得
            color_vector = image[i, j, :]

            # 色ベクトルを色平面に射影
            projected_vector = color_vector - np.dot(color_vector, u) * u

            # 射影された色ベクトルを保存
            projected_image[i, j, :] = projected_vector

    return projected_image

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

# 画像の読み込み
image = cv2.imread('/Users/hiyori/kang_plus_lab/images/map_c.ppm')
lab_image = rgb_to_lab(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

height, width, _ = image.shape
N = height * width

lab_process = np.zeros_like(image)
lab_uncripped = np.zeros_like(image)
lab_cripped = np.zeros_like(image)
lab_out = np.zeros_like(image)

u = angle_to_normal_vector(90 + 11.8)#2色覚平面

print("a")

project_image = project_pixels_to_color_plane(lab_image, u)

lab_cripped = clip_lab_within_rgb_gamut(project_image)

img_out_rgb = lab_to_rgb(lab_cripped)

# if img_out_rgb.dtype == np.float32 or img_out_rgb.dtype == np.float64:
#     # 最大値が1.0を超えない場合、255を掛ける
#     if img_out_rgb.max() <= 1.0:
#         img_out_rgb = (img_out_rgb * 255).astype(np.uint8)

# 射影された画像を表示
print("done!")

img_out_rgb = (img_out_rgb * 255).astype(np.uint8)

img_out_bgr = cv2.cvtColor(img_out_rgb, cv2.COLOR_RGB2BGR)

cv2.imwrite('/Users/hiyori/kang_plus_lab/images/map_pjt_plus.ppm',img_out_bgr)
cv2.imshow('lab_cripped', img_out_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
