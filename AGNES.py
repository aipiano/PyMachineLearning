import numpy as np
from numpy.linalg import linalg as la
from sklearn import datasets
from common import *
import matplotlib.pyplot as plt


class AGNES:
    def __init__(self):
        self.__labels = None
        self.__features = None
        self.__cluster_count = 0

    def train(self, sample_set: SampleSet, cluster_count):
        assert cluster_count < len(sample_set)
        features = sample_set.get_features_array()
        current_cluster_count = len(sample_set)
        # 每个样例所属的类别标签
        labels = array(range(current_cluster_count))
        # 合并子集后剩余的标签
        rest_labels = list(range(current_cluster_count))

        while current_cluster_count > cluster_count:
            min_set_distance = np.inf
            s1_label = 0
            s2_label = 0
            # 寻找距离最近的两个子集
            for i in range(len(rest_labels)):
                s1 = features[labels == rest_labels[i]]
                if len(s1) == 1:
                    s1 = s1.reshape((1, -1))

                for j in range(i + 1, len(rest_labels)):
                    s2 = features[labels == rest_labels[j]]
                    if len(s2) == 1:
                        s2 = s2.reshape((1, -1))

                    set_distance = AGNES.minimum_distance(s1, s2)
                    if set_distance >= min_set_distance:
                        continue
                    min_set_distance = set_distance
                    s1_label = rest_labels[i]
                    s2_label = rest_labels[j]

            # 合并距离最近的子集
            labels[labels == s2_label] = s1_label
            rest_labels.remove(s2_label)  # 合并后，删除该类别原来的标签
            current_cluster_count -= 1

        # 将标签调整为从0开始的连续整数
        for i, label in enumerate(rest_labels):
            labels[labels == label] = i

        self.__labels = labels
        self.__features = features
        self.__cluster_count = cluster_count

    def predict(self, sample_set: SampleSet):
        features = sample_set.get_features_array()
        cluster_count = self.__cluster_count
        labels = np.zeros((len(features),), dtype=int)

        clusters = []
        for i in range(cluster_count):
            clusters.append(self.__features[self.__labels == i])

        # 寻找与该样例距离最近的簇，并将该样例划为该簇对应的类别
        for i, feature in enumerate(features):
            f = feature.reshape((1, -1))
            min_distance = np.inf
            min_dist_label = 0
            for j, cluster in enumerate(clusters):
                distance = AGNES.minimum_distance(f, cluster)
                if distance < min_distance:
                    min_distance = distance
                    min_dist_label = j
            labels[i] = min_dist_label

        return labels

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

    @staticmethod
    def hausdorff(s1: np.ndarray, s2: np.ndarray):
        dm = AGNES.build_distance_matrix(s1, s2)
        d1 = np.max(np.min(dm, axis=0))
        d2 = np.max(np.min(dm, axis=1))
        return max(d1, d2)

    @staticmethod
    def average_distance(s1: np.ndarray, s2: np.ndarray):
        dm = AGNES.build_distance_matrix(s1, s2)
        tc = dm.shape[0] * dm.shape[1]
        return np.sum(dm) / tc

    @staticmethod
    def minimum_distance(s1: np.ndarray, s2: np.ndarray):
        dm = AGNES.build_distance_matrix(s1, s2)
        return np.min(dm)

    @staticmethod
    def build_distance_matrix(s1: np.ndarray, s2: np.ndarray):
        s1 = s1.reshape((s1.shape[0], 1, s1.shape[1]))
        s2 = s2.reshape((1, s2.shape[0], s2.shape[1]))
        return la.norm(s1 - s2, axis=2)


def test():

    n_samples = 400
    test_data = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)

    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    # colors = np.hstack([colors] * 20)

    samples = SampleSet()
    data_count = test_data[0].shape[0]
    for i in range(data_count):
        samples.append(Sample(list(test_data[0][i, :]), test_data[1][i]))

    '''
    距离的度量方式和数据的分布有极大关系，比如在环形数据分布下，最小距离度量可以完美聚类（JC = 1.0)
    而在一些其他分布下，最小距离度量的效果往往没有豪斯多夫距离好
    '''
    agnes = AGNES()
    agnes.train(samples, 2)
    labels = agnes.predict(samples)

    plt.scatter(test_data[0][:, 0], test_data[0][:, 1], s=10, c=colors[labels].tolist())
    plt.show()
    jc = agnes.score(samples)
    print("JC =", jc)

if __name__ == '__main__':
    test()