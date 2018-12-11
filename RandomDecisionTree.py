import numpy as np
from sklearn import datasets
from common import *
from DecisionTree import DecisionTree


class RandomDecisionTree(DecisionTree):
    """用于随机森林的决策树，不要单独使用"""

    @staticmethod
    def get_max_gain_split(samples, attributes: list):
        """
        获取纯度提升最大的划分，并返回增益率、用于划分的属性、以及划分后的各个子集
        """
        # 离散情况下，子集存在一个字典中，键是属性的取值，值是样本集合。
        # 连续情况下，子集只有两个，小于划分值的和大于划分值的，两个子集以及划分值保存在一个tuple中。

        k = int(np.log2(len(attributes)) + 0.5)
        if k == 0 and len(attributes) > 0:
            k = 1

        random_attributes = random.sample(attributes, k)
        return DecisionTree.get_max_gain_split(samples, random_attributes)


def test():
    test_data = datasets.load_iris()
    feature_types = [False, False, False, False]  # True: discrete; False: continuous

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
    random_decision_tree = RandomDecisionTree()
    random_decision_tree.train(train_set, attributes, 0)
    s = random_decision_tree.score(test_set)

    print(random_decision_tree)
    print('score =', s)

if __name__ == '__main__':
    test()

