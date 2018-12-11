"""
各种算法的公用数据结构
"""

import random
from numpy import array


class Sample:
    def __init__(self, _feature=None, _target=None):
        self.feature = _feature
        self.target = _target


class SampleSet:
    """
    保存所有训练数据的集合，只接受list类型的feature
    """
    def __init__(self, features=None, targets=None):
        if features and targets:
            self.__features = features
            self.__targets = targets
        else:
            self.__features = []
            self.__targets = []

    def append(self, sample):
        self.__features.append(sample.feature)
        self.__targets.append(sample.target)

    def remove(self, sample):
        if (sample.feature in self.__features) and (sample.target in self.__targets):
            self.__features.remove(sample.feature)
            self.__targets.remove(sample.target)
        else:
            assert 'no such sample'

    def get_features_array(self):
        return array(self.__features)

    def get_targets_array(self):
        return array(self.__targets)

    def get_features(self):
        return self.__features.copy()

    def get_targets(self):
        return self.__targets.copy()

    def __getitem__(self, key):
        if isinstance(key, slice):
            return SampleSet(self.__features.__getitem__(key), self.__targets.__getitem__(key))
        return Sample(self.__features[key], self.__targets[key])

    def __delitem__(self, key):
        self.__features.__delitem__(key)
        self.__targets.__delitem__(key)

    def __len__(self):
        return len(self.__features)

    def __str__(self):
        str_features = self.__features.__str__()
        str_targets = self.__targets.__str__()
        return 'features = %s\ntargets = %s' % (str_features, str_targets)


class Attribute:
    def __init__(self, _index=0, _discrete=True, _values=None):
        self.index = _index
        self.is_discrete = _discrete
        self.values = _values


def train_test_split(samples, test_ratio):
    set_size = len(samples)
    test_set_size = int(set_size * test_ratio + 0.5)
    train_set = samples[:]

    indices = random.sample(range(set_size), test_set_size)
    test_set = SampleSet()
    for idx in indices:
        train_set.remove(samples[idx])
        test_set.append(samples[idx])

    return train_set, test_set


def get_values_on_attribute(samples, attribute):
    """
    获取样本在指定属性上出现的所有取值
    """
    values = set()
    for sample in samples:
        values.add(sample.feature[attribute.index])

    # 若是连续属性，就进行排序
    if not attribute.is_discrete:
        values = sorted(values)
    return values


def count_class(samples):
    """
    统计当前样本集中每个类别的样本数
    """
    counts = {}
    for sample in samples:
        if not counts.get(sample.target):
            counts[sample.target] = 0
        counts[sample.target] += 1
    return counts


def bootstrap_sample(samples):
    """
    自助采样（需要分层采样吗？）
    :param samples:
    :return samples: SampleSet
    """
    result = SampleSet()
    count = len(samples)
    is_selected = [False] * count
    for i in range(count):
        idx = random.randrange(0, count)
        result.append(samples[idx])
        is_selected[idx] = True

    unselected_samples = SampleSet()
    for i in range(count):
        if not is_selected[i]:
            unselected_samples.append(samples[i])

    return result, unselected_samples
