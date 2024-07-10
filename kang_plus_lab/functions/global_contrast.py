import numpy as np
from sklearn.cluster import KMeans

def global_contrast(image, num_clusters=20):
    """
    画像のグローバルコントラストを計算

    :param image: CIE L*a*b* 色空間の入力画像。
    :param num_clusters: k-means クラスタリングで使用するクラスタの数。
    :return: グローバルコントラスト行列。
    """
    height, width, _ = image.shape
    N = height * width
    flat_image = image.reshape(N, 3)

    # k-means クラスタリングを適用
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(flat_image)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # 各クラスタペア間の色差ベクトルを計算
    Xg = []
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            color_diff = centroids[i] - centroids[j]
            Xg.append(color_diff)

    Xg = np.array(Xg)
    global_contrast_matrix = np.dot(Xg, Xg.T) / N

    return global_contrast_matrix