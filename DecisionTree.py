import numpy as np
from sklearn import datasets
from common import *
from BaseLeaner import BaseLeaner


class DecisionTree(BaseLeaner):
    def __init__(self):
        self.__tree = {}

    def train(self, sample_set: SampleSet, attributes: list, max_depth=0):
        """
        构建决策树，非递归方法
        """
        # 统计每个属性所有可能的取值
        for a in attributes:
            a.values = get_values_on_attribute(sample_set, a)

        # 创建节点
        root = {}
        node_stack = [root]
        param_stack = [(sample_set, attributes)]
        depth_stack = [0]  # 保存每个节点在树中的深度

        while len(param_stack) > 0:
            samples, attributes = param_stack.pop()
            node = node_stack.pop()
            depth = depth_stack.pop()

            # 如果节点的深度达到最大限度，提前返回
            if depth >= max_depth > 0:
                node['leaf'] = self.get_max_class(samples)
                continue

            # 若属性集为空，则当前节点标记为samples中样本数最多的类
            if len(attributes) == 0:
                node['leaf'] = self.get_max_class(samples)
                continue

            # 如果样本全部属于同一类，则将当前节点标记为该类，并返回
            target = samples[0].target
            all_in_one_class = True
            for sample in samples:
                if target != sample.target:
                    all_in_one_class = False
                    break
            if all_in_one_class:
                node['leaf'] = target
                continue

            # 选择最优的划分属性
            max_gain, split_attrib, splited_subsets = self.get_max_gain_split(samples, attributes)

            # 若样本集在当前属性集上无法划分, 则当前节点标记为samples中样本数最多的类
            if max_gain == 0.0 or not split_attrib:
                node['leaf'] = self.get_max_class(samples)
                continue

            # 当前节点还可以继续划分，键设为用于划分的属性
            node[split_attrib.index] = {}

            # 为划分属性每个可能的取值建立一个分支，并将之后需要处理的节点入栈
            if split_attrib.is_discrete:
                sub_attributes = attributes[:]
                sub_attributes.remove(split_attrib)
                for v in split_attrib.values:
                    if splited_subsets.get(v):
                        node[split_attrib.index][('=', v)] = {}
                        node_stack.append(node[split_attrib.index][('=', v)])
                        param_stack.append((splited_subsets[v], sub_attributes))
                        depth_stack.append(depth + 1)
            else:
                # 连续情况下，划分为两个分支
                subset_less_t = splited_subsets[0]
                subset_greater_t = splited_subsets[1]
                opt_t = splited_subsets[2]

                node[split_attrib.index][('<', opt_t)] = {}
                node_stack.append(node[split_attrib.index][('<', opt_t)])
                param_stack.append((subset_less_t, attributes))
                depth_stack.append(depth + 1)

                node[split_attrib.index][('>', opt_t)] = {}
                node_stack.append(node[split_attrib.index][('>', opt_t)])
                param_stack.append((subset_greater_t, attributes))
                depth_stack.append(depth + 1)

        self.__tree = root

    def predict(self, sample: Sample):
        """
        用训练好的决策树对sample进行预测
        """
        node = self.__tree
        while True:
            # 节点一定只包含一个键值对，键是用于划分的属性索引，值是进行划分的规则字典
            attrib_idx, rule_dict = next(iter(node.items()))

            # 如果是叶子节点，直接返回类型
            if attrib_idx == 'leaf':
                return rule_dict

            attrib_value = sample.feature[attrib_idx]

            rule_match = False
            # 规则是rule_dict的键，对应的值是下一个节点
            for rule, sub_node in rule_dict.items():
                # 每一个规则都是一个tuple，第一项是比较方法，第二项是比较对象
                if rule[0] == '=':
                    rule_match = attrib_value == rule[1]
                elif rule[0] == '<':
                    rule_match = attrib_value < rule[1]
                elif rule[0] == '>':
                    rule_match = attrib_value >= rule[1]

                if rule_match:
                    break

            if rule_match:
                assert sub_node is not None
                node = sub_node
            else:
                return None  # cannot predict

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

    def __str__(self):
        return self.__tree.__str__()

    @staticmethod
    def get_max_gain_split(samples, attributes):
        """
        获取纯度提升最大的划分，并返回增益率、用于划分的属性、以及划分后的各个子集
        """
        # 离散情况下，子集存在一个字典中，键是属性的取值，值是样本集合。
        # 连续情况下，子集只有两个，小于划分值的和大于划分值的，两个子集以及划分值保存在一个tuple中。

        gains = []
        gain_ratios = []
        split_attrib = []
        splited_subsets = []
        for attribute in attributes:
            if attribute.is_discrete:
                gain, gain_ratio, subsets = DecisionTree.calc_gain_discrete(samples, attribute)
            else:
                gain, gain_ratio, subsets = DecisionTree.calc_gain_continuous(samples, attribute)

            gains.append(gain)
            gain_ratios.append(gain_ratio)
            split_attrib.append(attribute)
            splited_subsets.append(subsets)

        gains = array(gains)
        gain_mean = np.mean(gains)

        max_gain_ratio = 0.0
        idx = 0
        # 从划分的属性中找出增益大于平均的，再从中找出增益率最大的
        for i, gain in enumerate(gains):
            if gain < gain_mean:
                continue
            gain_ratio = gain_ratios[i]
            if gain_ratio > max_gain_ratio:
                max_gain_ratio = gain_ratio
                idx = i

        return max_gain_ratio, split_attrib[idx], splited_subsets[idx]

    @staticmethod
    def get_max_class(samples):
        """
        返回样本集中样本数最多的类别
        """
        counts = count_class(samples)
        max_count = 0
        target = samples[0].target
        for t, c in counts.items():
            if c > max_count:
                max_count = c
                target = t
        return target

    @staticmethod
    def entropy(samples):
        """
        计算样本集的信息熵
        """
        counts = count_class(samples)
        sample_count = float(len(samples))
        ent = 0.0
        for c in counts.values():
            p = c / sample_count
            ent -= p * np.log2(p)
        return ent

    @staticmethod
    def calc_gain_discrete(samples, attribute):
        """
        计算samples以attribute划分后的信息增益，适用于离散属性
        """
        ent_samples = DecisionTree.entropy(samples)

        # 获取在attribute对应的属性上，取值为key的样本集
        attrib_idx = attribute.index
        dic_subsets = {}
        for sample in samples:
            key = sample.feature[attrib_idx]
            if not dic_subsets.get(key):
                dic_subsets[key] = []
            dic_subsets[key].append(sample)

        # 计算所有子集信息熵的和
        set_size = float(len(samples))
        sum_entropy = 0.0
        iv = 0.0
        for subset in dic_subsets.values():
            subset_size = float(len(subset))
            size_ratio = subset_size / set_size
            sum_entropy += size_ratio * DecisionTree.entropy(subset)
            iv -= size_ratio * np.log2(size_ratio)

        # 返回增益
        gain = ent_samples - sum_entropy
        if gain == 0.0:
            gain_ratio = 0.0
        else:
            gain_ratio = gain / iv
        return gain, gain_ratio, dic_subsets

    @staticmethod
    def calc_gain_continuous(samples, attribute):
        """
        计算samples以attribute划分后的信息增益，适用于连续属性
        """
        ent_samples = DecisionTree.entropy(samples)

        # 获取在attribute对应的属性上，取值为key的样本集
        attrib_idx = attribute.index
        attrib_values = get_values_on_attribute(samples, attribute)

        # 样本在该属性上的取值都相同，无法划分，增益为0
        if len(attrib_values) < 2:
            return 0.0, 0.0, None

        set_size = float(len(samples))
        min_ent = 100000.0
        iv = 0.0
        opt_t = 0.5 * (attrib_values[0] + attrib_values[1])
        subset_less_t = []
        subset_greater_t = []
        for i in range(len(attrib_values) - 1):
            greater_t = []
            less_t = []
            t = 0.5 * (attrib_values[i] + attrib_values[i+1])
            for sample in samples:
                if sample.feature[attrib_idx] < t:
                    less_t.append(sample)
                else:
                    greater_t.append(sample)

            w1 = len(less_t) / set_size
            w2 = len(greater_t) / set_size
            ent = w1 * DecisionTree.entropy(less_t) + w2 * DecisionTree.entropy(greater_t)
            if ent < min_ent:
                min_ent = ent
                iv = - w1 * np.log2(w1) - w2 * np.log2(w2)
                opt_t = t
                subset_less_t = less_t
                subset_greater_t = greater_t

        gain = ent_samples - min_ent
        if gain == 0.0:
            gain_ratio = 0.0
        else:
            gain_ratio = gain / iv
        return gain, gain_ratio, (subset_less_t, subset_greater_t, opt_t)


def build_tree(samples, attributes):
    """
    构建决策树，递归方法
    """
    # 如果样本全部属于同一类，则将当前节点标记为该类，并返回
    target = samples[0].target
    all_in_one_class = True
    for sample in samples:
        if target != sample.target:
            all_in_one_class = False
            break
    if all_in_one_class:
        return {'leaf': target}

    # 若属性集为空，则当前节点标记为samples中样本数最多的类
    if len(attributes) == 0:
        return {'leaf': DecisionTree.get_max_class(samples)}

    # 选择最优的划分属性
    max_gain, split_attrib, splited_subsets = DecisionTree.get_max_gain_split(samples, attributes)

    # 若样本集在当前属性集上无法划分, 则当前节点标记为samples中样本数最多的类
    if max_gain == 0.0 or not split_attrib:
        return {'leaf': DecisionTree.get_max_class(samples)}

    # 创建节点
    node = {split_attrib.index: {}}

    # 为划分属性每个可能的取值建立一个分支，递归建立树节点
    if split_attrib.is_discrete:
        sub_attributes = attributes[:]
        sub_attributes.remove(split_attrib)
        for v in split_attrib.values:
            if splited_subsets.get(v):
                node[split_attrib.index][('=', v)] = build_tree(splited_subsets[v], sub_attributes)
    else:
        # 连续情况下，划分为两个分支
        subset_less_t = splited_subsets[0]
        subset_greater_t = splited_subsets[1]
        opt_t = splited_subsets[2]
        node[split_attrib.index][('<', opt_t)] = build_tree(subset_less_t, attributes)
        node[split_attrib.index][('>', opt_t)] = build_tree(subset_greater_t, attributes)

    return node


def test():
    test_data = datasets.load_iris()
    feature_types = [False] * 4  # True: discrete; False: continuous

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
    decision_tree = DecisionTree()
    decision_tree.train(train_set, attributes, 5)
    s = decision_tree.score(test_set)

    print(decision_tree)
    print('score =', s)

if __name__ == '__main__':
    test()


