import numpy as np
from numpy.linalg import linalg as la
from sklearn import datasets
from common import *
import matplotlib.pyplot as plt
"""
基于Science上发表的Clustering by fast search and find of density peaks.
该算法虽然十分高效、巧妙，但对于没有明显聚类中心的数据分布效果不好，比如环形数据集。有待改进。
该算法虽然在无明显聚类中心的数据分布中效果不好，但可以借用半监督聚类的机制，达到很好的聚类效果
"""


class DensityGraphClustering:
    def __init__(self):
        self.__centers = None
        self.__features = None
        self.__labels = None
        self.__densities = None

    def train(self, sample_set: SampleSet, cluster_count=0, cutoff_ratio=0.1):
        assert cluster_count < len(sample_set)
        features = sample_set.get_features_array()

        distance_matrix = self.build_distance_matrix(features, features)
        cutoff_distance = self.estimate_cutoff_distance(distance_matrix, cutoff_ratio)
        densities = self.calc_densities(distance_matrix, cutoff_distance)

        # 根据每个点的密度，进行升序排列
        sort_indices = np.argsort(densities)
        features = features[sort_indices]
        densities = densities[sort_indices]
        # 先对行进行排序，再对列进行排序，就得到排序后的距离矩阵
        distance_matrix = distance_matrix[sort_indices][:, sort_indices]

        distances, neighbor_indices = self.calc_distances_and_neighbors(distance_matrix, densities)

        # 归一化距离与密度并计算乘积
        densities /= np.max(densities)
        distances /= np.max(distances)
        gamas = densities * distances

        # 降序排列
        center_indices = np.argsort(gamas)[-1::-1]
        gamas = gamas[center_indices]

        plt.scatter(range(len(gamas)), gamas)
        plt.show()

        if cluster_count == 0:
            # 求gama最大的前百分之十的均值，加上0.1偏移作为判断聚类中心的阈值
            threshold = np.mean(gamas[:len(gamas) * 0.1]) + 0.1
            cluster_count = len(gamas[gamas > threshold])

        self.__centers = features[center_indices[:cluster_count]]

        labels = np.zeros((len(features), ), dtype=int)
        labels[:] = -1
        # 先对所有聚类中心赋予标签
        labels[center_indices[:cluster_count]] = array(range(cluster_count))

        # 倒序遍历，从密度最大的点（一定是距离中心且被标记）开始，传播到密度较小的点
        for i in range(len(labels) - 1, -1, -1):
            if labels[i] >= 0:
                continue
            # 若距离邻居的距离过大，索引就是-1，则不对该样例分类
            if neighbor_indices[i] >= 0:
                labels[i] = labels[neighbor_indices[i]]

        self.__labels = labels
        self.__features = features
        self.__densities = densities
        return cluster_count

    def predict(self, sample_set: SampleSet):
        pass

    def score(self, sample_set: SampleSet):
        targets = sample_set.get_targets()
        labels = self.predict(sample_set)

        ss = 0
        sd = 0
        ds = 0
        dd = 0
        m = len(labels)
        for i in range(m):
            for j in range(i + 1, m):
                if labels[i] == labels[j] and targets[i] == targets[j]:
                    ss += 1
                elif labels[i] == labels[j] and targets[i] != targets[j]:
                    sd += 1
                elif labels[i] != labels[j] and targets[i] == targets[j]:
                    ds += 1
                elif labels[i] != labels[j] and targets[i] != targets[j]:
                    dd += 1

        jc = ss / (ss + sd + ds)
        # fmi = ss / np.sqrt((ss + sd) * (ss + ds))
        # ri = 2 * (ss + dd) / (m * (m - 1))

        # Jaccard系数（JC），[0, 1]之间，越大越好
        return jc

    def get_cluster_centers(self):
        return self.__centers

    def get_labels(self):
        return self.__labels

    def get_features(self):
        return self.__features

    @staticmethod
    def calc_densities(distance_matrix: np.ndarray, cutoff_distance):
        cutoff_dm = distance_matrix / cutoff_distance
        cutoff_dm **= 2
        cutoff_dm *= -1
        exp_dm = np.exp(cutoff_dm)
        return np.sum(exp_dm, axis=1) - 1.0

    @staticmethod
    def calc_distances_and_neighbors(distance_matrix: np.ndarray, densities: np.ndarray):
        """
        求取每个点到密度比它大的点的最小距离，以及距离最小点的索引
        :param distance_matrix:
        :param densities:
        :return:
        """
        distances = np.zeros((len(densities), ), dtype=float)
        neighbors = np.zeros((len(distances), ), dtype=int)
        # 计算每个点到密度比它大的点的最小距离（排除最后一行，即密度最大的点）
        for i, r in enumerate(distance_matrix[:-1]):
            idx = np.argmin(r[i + 1:]) + i + 1
            distances[i] = r[idx]
            neighbors[i] = idx

        # 计算密度最大的点的距离
        distances[-1] = np.max(distances[:-1])
        neighbors[-1] = -1
        return distances, neighbors

    @staticmethod
    def build_distance_matrix(s1: np.ndarray, s2: np.ndarray):
        s1 = s1.reshape((s1.shape[0], 1, s1.shape[1]))
        s2 = s2.reshape((1, s2.shape[0], s2.shape[1]))
        return la.norm(s1 - s2, axis=2)

    @staticmethod
    def estimate_cutoff_distance(distance_matrix: np.ndarray, cutoff_ratio):
        nonzero_dm = np.tril(distance_matrix).flatten()
        nonzero_dm = nonzero_dm[nonzero_dm > 0]
        # 对所有非零距离值排序
        nonzero_dm = np.sort(nonzero_dm)
        return nonzero_dm[int(cutoff_ratio * len(nonzero_dm) + 0.5)]


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

    dgc = DensityGraphClustering()
    dgc.train(samples)
    # labels = dgc.predict(samples)
    labels = dgc.get_labels()
    features = dgc.get_features()
    centers = dgc.get_cluster_centers()

    '''
    该算法虽然十分高效、巧妙，但对于没有明显聚类中心的数据分布效果不好，比如环形数据集。有待改进
    '''
    plt.scatter(features[:, 0], features[:, 1], s=20, c=colors[labels])
    plt.scatter(centers[:, 0], centers[:, 1], s=100)
    plt.show()
    # jc = dgc.score(samples)
    # print("JC =", jc)

if __name__ == '__main__':
    test()
