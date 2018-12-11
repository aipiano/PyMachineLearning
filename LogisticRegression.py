import numpy as np
from sklearn import datasets
from scipy.optimize import minimize
from common import *
from BaseLeaner import BaseLeaner


class LogisticRegression(BaseLeaner):
    """
    对数几率回归，只能用于二分类问题
    """
    def __init__(self):
        self.__w = array([])
        self.__b = 0.0

    def train(self, sample_set: SampleSet, attributes: list):
        xs = sample_set.get_features_array()
        ys = sample_set.get_targets_array()
        one_col = np.ones((xs.shape[0], 1), dtype=xs.dtype)
        # 转至，方便之后的优化算法进行内积运算
        xs = np.hstack((xs, one_col)).T

        w0 = np.zeros((xs.shape[0],), dtype=xs.dtype)
        result = minimize(self.log_likelihood, w0,
                          args=(xs, ys), method='Newton-CG',
                          jac=self.jac, hess=self.hess)
        self.__w = result.x[:-1]
        self.__b = result.x[-1]

    def predict(self, sample: Sample):
        feature = array(sample.feature)
        y = self.__w.dot(feature) + self.__b
        return int(y > 0.5)

    def score(self, sample_set):
        true_count = 0
        for sample in sample_set:
            result = self.predict(sample)
            if result == sample.target:
                true_count += 1

        return float(true_count) / len(sample_set)

    @staticmethod
    def log_likelihood(w, *args):
        """
        用于优化的目标函数（对数似然函数）
        :param w:
        :param args:
        :return:
        """
        xs, ys = args
        w_xs = w.dot(xs)
        log_probs = -ys * w_xs + np.log(1 + np.exp(w_xs))
        return np.sum(log_probs)

    @staticmethod
    def jac(w, *args):
        """
        目标函数的一阶导数
        :param w:
        :param args:
        :return:
        """
        xs, ys = args
        e_w_xs = np.exp(w.dot(xs))
        p = e_w_xs / (1 + e_w_xs)
        return np.sum(xs * (p - ys), axis=1)

    @staticmethod
    def hess(w, *args):
        """
        目标函数的二阶导数
        :param w:
        :param args:
        :return:
        """
        xs, ys = args
        e_w_xs = np.exp(w.dot(xs))
        p = e_w_xs / (1 + e_w_xs)
        pp = p * (1 - p)

        rows = xs.shape[0]
        xi = xs[:, 0].reshape((rows, 1))
        h = xi @ xi.T
        for i in range(xs.shape[1]):
            xi = xs[:, i].reshape((rows, 1))
            h += pp[i] * xi @ xi.T
        return h


def test():
    test_data = datasets.load_breast_cancer()
    feature_types = [False] * 30  # True: discrete; False: continuous

    samples = SampleSet()
    attributes = []
    data_count = test_data.data.shape[0]
    attrib_count = test_data.data.shape[1]

    for i in range(data_count):
        samples.append(Sample(list(test_data.data[i, :]), test_data.target[i]))

    for i in range(attrib_count):
        a = Attribute(i, feature_types[i])
        attributes.append(a)

    train_set, test_set = train_test_split(samples, 0.3)
    logistic_regression = LogisticRegression()
    logistic_regression.train(train_set, attributes)
    s = logistic_regression.score(test_set)

    print('score =', s)

if __name__ == '__main__':
    test()