import numpy as np
from numpy.linalg import linalg as la
from sklearn import datasets
from common import *
import matplotlib.pyplot as plt


class Kmeans:
    def __init__(self):
        self.__centers = None
        self.__labels = None

    def train(self, sample_set: SampleSet, cluster_count, max_iter_count):
        assert cluster_count < len(sample_set)
        if isinstance(sample_set, SampleSet):
            features = sample_set.get_features_array()
        else:
            features = sample_set
        # 保存每个样本的类标记
        labels = np.zeros((features.shape[0],), dtype=int)
        centers = []

        # 从样本中随机采样cluster_count个样例作为簇的初始中心
        center_indices = random.sample(range(len(sample_set)), cluster_count)
        for i in center_indices:
            centers.append(features[i, :])

        # 保存每个样本到距离最近的中心的距离
        min_dists = np.zeros(labels.shape, dtype=np.float32)

        updated = True
        iter_count = 0
        while updated and iter_count <= max_iter_count:
            iter_count += 1
            # 更新分类标记
            min_dists[:] = np.inf
            for i in range(len(centers)):
                dists = la.norm(features - centers[i], axis=1)
                # 一开始一定会将labels全部设置为0
                labels[dists < min_dists] = i
                min_dists = np.minimum(dists, min_dists)

            # 更新类中心
            updated = False
            for i in range(len(centers)):
                fi = features[labels == i]
                new_center = np.sum(fi, axis=0) / fi.shape[0]
                if not np.allclose(centers[i], new_center):
                    centers[i][:] = new_center
                    updated = True

        self.__centers = array(centers)
        self.__labels = labels
        return iter_count

    def predict(self, sample_set: SampleSet):
        features = sample_set.get_features_array()
        centers = self.__centers

        labels = np.zeros((features.shape[0],), dtype=int)
        min_dists = np.zeros(labels.shape, dtype=np.float32)
        min_dists[:] = np.inf
        for i in range(len(centers)):
            dists = la.norm(features - centers[i], axis=1)
            # 一开始一定会将labels全部设置为0
            labels[dists < min_dists] = i
            min_dists = np.minimum(dists, min_dists)

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

    def get_centers(self):
        return self.__centers

    def get_labels(self):
        return self.__labels


def test():
    n_samples = 1500
    test_data = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)

    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    # colors = np.hstack([colors] * 20)

    samples = SampleSet()
    data_count = test_data[0].shape[0]
    for i in range(data_count):
        samples.append(Sample(list(test_data[0][i, :]), test_data[1][i]))

    kmeans = Kmeans()
    iter_count = kmeans.train(samples, 3, 100)
    jc = kmeans.score(samples)

    labels = kmeans.predict(samples)
    plt.scatter(test_data[0][:, 0], test_data[0][:, 1], s=10, c=colors[labels])
    plt.show()

    print(kmeans.get_centers())
    print("Iterate Count:", iter_count)
    print("JC =", jc)

if __name__ == '__main__':
    test()