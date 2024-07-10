import numpy as np
import cv2
from functions import *
from scipy.optimize import minimize
import sys

def main(image_path, apply_weight=True):
    image = cv2.imread(image_path)
    assert image is not None, "読み込みに失敗しました"

    height, width, _ = image.shape
    N = height * width

    # G(u)を計算する関数、今回は未搭載
    # g = global_contrast.global_contrast(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))

    #TODO J(u)を計算する関数、今回は未搭載
    # j = alpha * l + (1 - alpha) * g

    #Xlを計算する関数
    Xl, w = calculate_color_difference_vectors_with_gaussian_pairing.calculate_color_difference_vectors_with_gaussian_pairing(image)

    if apply_weight:
        Xl_weight = np.zeros_like(Xl)
        for i in range(N):
            Xl_weight[i, 0] = Xl[i, 0] * w[i]
            Xl_weight[i, 1] = Xl[i, 1] * w[i]
            Xl_weight[i, 2] = Xl[i, 2] * w[i]
        Al_new = Xl_weight.T @ Xl_weight
    else:
        Al_new = Xl.T @ Xl

    # 目的関数と制約条件の定義
    def objective(u):
        return ((u.T @ Al_new @ u) / N)

    # L*軸に垂直な制約
    def constraint_perpendicular_to_L_star(u):
        e = np.array([1, 0, 0])  # L*軸を指す標準基底ベクトル
        return np.dot(u, e)

    # 単位ベクトル制約
    def constraint_unit_vector(u):
        return np.dot(u, u) - 1

    # 初期値の設定
    u0 = np.random.rand(3)
    u0 /= np.linalg.norm(u0)  # 単位ベクトル化

    # 制約の設定
    constraints = [{'type': 'eq', 'fun': constraint_perpendicular_to_L_star},
                   {'type': 'eq', 'fun': constraint_unit_vector}]

    # 逐次二次計画法による最適化問題の解決
    res = minimize(objective, u0, constraints=constraints, method="SLSQP")

    # 最適化された u の値
    optimized_u = res.x
    optimized_u = np.reshape(optimized_u, (3, 1))

    # 最適化された結果の u^T Al u の値を計算
    optimized_value = (optimized_u.T @ Al_new @ optimized_u) / N

    x = optimized_u.reshape(1, 3)[0, 1]
    y = optimized_u.reshape(1, 3)[0, 2]

    optimized_degree = (101.48 - (np.rad2deg(np.arctan2(y, x)) + 90)) % 180

    weight_status = "あり" if apply_weight else "なし"
    print(f"画像名: {image_path}, 明度差重み: {weight_status}, 最適回転角度: {optimized_degree}")

if __name__ == "__main__":
    image_path = 'images/flower.ppm'
    user_input = input("明度差重みをつけますか？ (y/n): ").strip().lower()
    if user_input == "y":
        apply_weight = True
    elif user_input == "n":
        apply_weight = False
    else:
        print("無効な入力です。プログラムを終了します。")
        sys.exit(1)  # プログラムを終了

    main(image_path, apply_weight)

print("done!")
