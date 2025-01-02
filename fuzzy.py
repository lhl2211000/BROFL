import numpy as np
import skfuzzy as fuzz

# 构造特征矩阵（归一化后的特征）
data = np.array([
    [0.1, 0.2, 0.1],  # 客户端 1
    [0.2, 0.1, 0.3],  # 客户端 2
    [0.4, 0.2, 0.3],  # 客户端 3
    [0.2, 0.3, 0.4],  # 客户端 4
    [0.8, 0.7, 0.8],  # 客户端 5
    [0.4, 0.6, 0.5],  # 客户端 6
])

# FCM聚类
n_clusters = 2  # 设定簇的数量为2
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data.T, n_clusters, m=2, error=0.005, maxiter=1000, init=None
)

# 打印隶属度矩阵
print("隶属度矩阵 (u):")
print(u)

# 打印簇中心
print("\n簇中心 (cntr):")
print(cntr)

