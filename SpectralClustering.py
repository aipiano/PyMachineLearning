import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla
from sklearn import datasets
from common import *
import matplotlib.pyplot as plt
from Kmeans import Kmeans
"""
谱聚类，使用归一化的拉普拉斯矩阵，欧氏距离，KNN权重
谱聚类真优雅！！！
"""


class SpectralClustering:
    def __init__(self):
        self.__features = None
        self.__kmeans = Kmeans()

    def train(self, sample_set: SampleSet, knn_k=0, cluster_count=0):
        assert cluster_count < len(sample_set)
        features = sample_set.get_features_array()
        if knn_k <= 0:
            knn_k = int(np.log(len(features)) + 1)

        L = self.build_laplacian_matrix(features, knn_k)
        # 最大聚类数设为样本数的百分之十，即每个类别平均10个样例
        max_cluster_count = int(0.1 * len(L) + 0.5)
        eig_values, eig_vectors = sla.eigs(L, k=max_cluster_count, which='SR')
        eig_values = np.real(eig_values)
        eig_vectors = np.real(eig_vectors)

        sort_idx = np.argsort(eig_values)
        eig_values = eig_values[sort_idx]
        # 特征向量以列保存，所以对列进行排序
        eig_vectors = eig_vectors[:, sort_idx]

        if cluster_count <= 0:
            cluster_count = 1
            # 由eigen gap来确定聚类数目
            for i in range(1, len(eig_values)):
                if np.isclose(eig_values[i], 0):
                    continue
                if eig_values[i - 1] == 0:
                    cluster_count = i
                    break
                if eig_values[i] > 1 or i >= max_cluster_count:   # 特征值或聚类数过大，提前返回
                    cluster_count = i
                    break
                if abs(eig_values[i] / eig_values[i - 1]) > 1e8:
                    cluster_count = i
                    break

        self.__kmeans.train(eig_vectors[:, 0:cluster_count], cluster_count, 100)
        self.__features = features

    def get_labels(self):
        return self.__kmeans.get_labels()

    def get_features(self):
        return self.__features

    @staticmethod
    def build_laplacian_matrix(s: np.ndarray, knn_k):
        """
        计算归一化拉普拉斯矩阵，相似图采用KNN图
        :param s:
        :param knn_k:
        :return:
        """
        s1 = s.reshape((s.shape[0], 1, s.shape[1]))
        s2 = s.reshape((1, s.shape[0], s.shape[1]))
        dm = la.norm(s1 - s2, axis=2)

        knn_indices = np.argsort(dm, axis=1)[:, :knn_k + 1]
        W = np.zeros(dm.shape, dm.dtype)
        for i in range(len(W)):
            # 直接用1表示相连，0表示勿连，也可以使用带权重的连接表示（距离越近，权重越大）
            W[knn_indices[i], i] = W[i, knn_indices[i]] = 1

        degrees = np.sum(W, axis=1)
        D = np.diag(degrees)

        return np.eye(len(D), dtype=D.dtype) - la.inv(D) @ W


def test():
    n_samples = 1500
    # test_data = datasets.make_moons(n_samples=n_samples, noise=.05)
    test_data = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=.05)
    # test_data = datasets.make_blobs(n_samples=n_samples, random_state=8)
    # test_data = np.random.rand(n_samples, 2), np.zeros((n_samples, ), dtype=int)

    # test_data = datasets.load_iris()

    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)

    samples = SampleSet()
    data_count = test_data[0].shape[0]
    for i in range(data_count):
        samples.append(Sample(list(test_data[0][i, :]), test_data[1][i]))

    sc = SpectralClustering()
    sc.train(samples, cluster_count=0)
    # labels = dgc.predict(samples)
    labels = sc.get_labels()
    features = sc.get_features()

    plt.scatter(features[:, 0], features[:, 1], s=20, c=colors[labels])
    plt.show()
    # jc = dgc.score(samples)
    # print("JC =", jc)

if __name__ == '__main__':
    test()