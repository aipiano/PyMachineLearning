import numpy as np
from sklearn import datasets
from common import *
from BaseLeaner import BaseLeaner


class NaiveBayes(BaseLeaner):
    def __init__(self):
        self.__pc = {}
        self.__pxc = {}
        self.__feature_length = 0

    def train(self, sample_set: SampleSet, attributes: list):
        set_size = len(sample_set)
        self.__feature_length = len(attributes)
        assert self.__feature_length == len(sample_set[0].feature)

        features = sample_set.get_features_array()
        targets = sample_set.get_targets_array()

        # 统计每个离散属性所有可能的取值
        for a in attributes:
            if a.is_discrete:
                a.values = set(features[:, a.index])

        counts = count_class(sample_set)
        class_count = len(counts.keys())
        for t, c in counts.items():
            # 拉普拉斯修正
            self.__pc[t] = (c + 1) / (set_size + class_count)

            # 事先分配好数列元素的个数
            self.__pxc[t] = [0] * self.__feature_length
            target_is_t = targets == t
            for a in attributes:
                i = a.index  # 属性对应的列索引
                attrib_values = features[:, i]  # 该属性的所有样本值
                if a.is_discrete:
                    value_count = len(a.values)
                    self.__pxc[t][i] = {}
                    for v in a.values:
                        value_is_v = attrib_values == v
                        count_tv = np.count_nonzero(target_is_t & value_is_v)
                        # 属于t类型的样本中，第i个属性取值为v的概率
                        self.__pxc[t][i][v] = (count_tv + 1) / (c + value_count)
                else:
                    attrib_values_in_class_t = attrib_values[target_is_t]
                    mean = np.mean(attrib_values_in_class_t)
                    var = np.var(attrib_values_in_class_t)
                    # 属于t类型的样本中，第i个属性的均值和方差
                    self.__pxc[t][i] = (mean, var)

    def predict(self, sample: Sample):
        probs = {}
        # 计算样本属于每个类别的概率
        for t, pc in self.__pc.items():
            probs[t] = pc
            for i in range(self.__feature_length):
                if isinstance(self.__pxc[t][i], dict):
                    probs[t] += self.__pxc[t][i][sample.feature[i]]
                else:
                    probs[t] += self.get_normal_dist_density(sample.feature[i], *self.__pxc[t][i])

        max_prob = 0.0
        opt_class = None
        # 以概率最大的类别作为分类结果
        for t, p in probs.items():
            if p > max_prob:
                max_prob = p
                opt_class = t
        return opt_class

    def score(self, sample_set: SampleSet):
        """
        评估决策树在所给样本集上的正确率
        """
        true_count = 0
        for sample in sample_set:
            result = self.predict(sample)
            if result == sample.target:
                true_count += 1

        return float(true_count) / len(sample_set)

    @staticmethod
    def get_normal_dist_density(x, mean, var):
        return (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-(x - mean) ** 2 / (2 * var))


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
    naive_bayes = NaiveBayes()
    naive_bayes.train(train_set, attributes)
    s = naive_bayes.score(test_set)

    print('score =', s)

if __name__ == '__main__':
    test()
