import functions

def calculate_J(u, Xl, Xg, W, alpha):
    """
    ローカルとグローバルコントラストを組み合わせてJ(u)を計算します。
    alpha: 0と1の間の重み付け係数。
    """
    L_u = calculate_L(u, Xl)
    G_u = calculate_G(u, Xg, W)
    return alpha * L_u + (1 - alpha) * G_u