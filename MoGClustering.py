import numpy as np
from numpy.linalg import linalg as la
from sklearn import datasets
from common import *
import matplotlib.pyplot as plt


class MoGClustering:
    def __init__(self):
        self.__a = None
        self.__u = None
        self.__sigma = None

    def train(self, sample_set: SampleSet, cluster_count, max_iter_count):
        assert cluster_count < len(sample_set)
        features = sample_set.get_features_array()

        # 初始化混合高斯参数
        a = np.ones((cluster_count,), dtype=np.float32)
        a /= cluster_count  # 所有类型的初始权重相同

        indices = random.sample(range(len(sample_set)), cluster_count)
        u = np.zeros((cluster_count, features.shape[1]), dtype=float)
        for i in range(cluster_count):
            # 从样本中随机采样cluster_count个样例作为簇的初始中心
            u[i] = features[indices[i]]

        sigma = np.zeros((cluster_count, u.shape[1], u.shape[1]), dtype=float)
        for i in range(cluster_count):
            # 每个协方差矩阵初始化为对角元素为0.1的对角阵
            sigma[i, :, :] = np.eye(u.shape[1], dtype=u.dtype) * 0.1

        # gama[i, j]代表第j个样例属于第i类的概率
        gama = np.zeros((cluster_count, len(sample_set)), dtype=float)
        prob_sum = np.zeros((len(sample_set),), dtype=float)
        iter_count = 0
        last_likelihood = -np.inf

        # 最大期望算法（EM），开始迭代
        while iter_count <= max_iter_count:
            # E步
            for i in range(cluster_count):
                gama[i] = a[i] * MoGClustering.probs(features, u[i], sigma[i])
            prob_sum[:] = np.sum(gama, axis=0)

            assert 0.0 not in prob_sum

            # 如果似然增长过小，提前结束循环
            likelihood = np.sum(np.log(prob_sum))
            if np.isclose(last_likelihood, likelihood):
                break

            last_likelihood = likelihood
            gama /= prob_sum

            # M步
            # 更新权重
            a = np.sum(gama, axis=1)
            u[:] = 0
            sigma[:, :, :] = 0
            # 更新均值
            for i in range(cluster_count):
                for j in range(len(sample_set)):
                    u[i] += gama[i, j] * features[j]
                u[i] /= a[i]

            # 更新协方差矩阵
            for i in range(cluster_count):
                for j in range(len(sample_set)):
                    dx = (features[j] - u[i]).reshape((-1, 1))
                    sigma[i] += gama[i, j] * dx @ dx.T
                sigma[i] /= a[i]

            # 最后归一化权重
            a /= len(features)

            iter_count += 1

        self.__a = a
        self.__u = u
        self.__sigma = sigma
        return iter_count

    def predict(self, sample_set: SampleSet):
        features = sample_set.get_features_array()
        cluster_count = len(self.__u)

        # gama[i, j]代表第j个样例属于第i类的概率
        gama = np.zeros((cluster_count, len(sample_set)), dtype=float)
        prob_sum = np.zeros((len(sample_set),), dtype=float)

        for i in range(cluster_count):
            gama[i] = self.__a[i] * MoGClustering.probs(features, self.__u[i], self.__sigma[i])
        prob_sum[:] = np.sum(gama, axis=0)
        gama /= prob_sum

        labels = np.argmax(gama, axis=0)
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
    def probs(x, u, sigma):
        """
        求x在以均值为u，协方差为sigma的多元高斯分布中的概率密度
        :param x: 随机变量，每一行是一个特征向量
        :param u: 均值
        :param sigma: 协方差矩阵
        :return:
        """
        dx = x - u
        dx_div_sigma = dx @ la.inv(sigma)
        p = np.zeros((x.shape[0], ), dtype=dx.dtype)
        for i in range(len(p)):
            p[i] = np.dot(dx_div_sigma[i], dx[i])
        p *= -0.5

        n = sigma.shape[0]
        d = 1.0 / np.sqrt((2 * np.pi)**n * la.det(sigma))
        return np.exp(p) * d


def test():
    n_samples = 1500
    test_data = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)

    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    # colors = np.hstack([colors] * 20)

    samples = SampleSet()
    data_count = test_data[0].shape[0]
    for i in range(data_count):
        samples.append(Sample(list(test_data[0][i, :]), test_data[1][i]))

    '''
    选用不同的距离度量方式，最小距离和平均距离的效果一般，豪斯多夫距离的效果优于前两者
    '''
    mog = MoGClustering()
    iter_count = mog.train(samples, 3, 100)
    jc = mog.score(samples)

    labels = mog.predict(samples)
    plt.scatter(test_data[0][:, 0], test_data[0][:, 1], s=10, c=colors[labels])
    plt.show()

    '''
    与K-means相比，混合高斯聚类所需的迭代次数稍多，但结果往往也更好
    '''
    print("Iterate Count:", iter_count)
    print("JC =", jc)

if __name__ == '__main__':
    test()