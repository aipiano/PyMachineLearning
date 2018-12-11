from sklearn import datasets
from common import *
from BaseLeaner import BaseLeaner
from RandomDecisionTree import RandomDecisionTree
from NaiveBayes import NaiveBayes
from LogisticRegression import LogisticRegression


class Bagging:
    def __init__(self, base_leaner_count, base_leaner: BaseLeaner):
        self.__leaner_count = base_leaner_count
        self.__base_learner = base_leaner
        self.__leaners = []

    def train(self, sample_set: SampleSet, attributes: list, *args, **kwargs):
        self.__leaners.clear()
        for i in range(self.__leaner_count):
            random_samples, rest_samples = bootstrap_sample(sample_set)
            new_leaner = self.__base_learner.new()
            new_leaner.train(random_samples, attributes, *args, **kwargs)
            # 进行包外估计，只保留正确率大于50%的学习器
            if new_leaner.score(rest_samples) > 0.5:
                self.__leaners.append(new_leaner)

    def predict(self, sample: Sample):
        """
        相对多数投票法
        :param sample:
        :return:
        """
        votings = {}
        for leaner in self.__leaners:
            target = leaner.predict(sample)
            if target not in votings:
                votings[target] = 0
            votings[target] += 1

        final_target = None
        max_vote = 0
        for t, vote in votings.items():
            if vote < max_vote:
                continue
            max_vote = vote
            final_target = t

        return final_target

    def score(self, sample_set: SampleSet):
        """
        评估决策树在所给样本集上的正确率
        """
        for i, leaner in enumerate(self.__leaners):
            print('leaner', i, leaner.score(sample_set))

        true_count = 0
        for sample in sample_set:
            result = self.predict(sample)
            if result == sample.target:
                true_count += 1

        return float(true_count) / len(sample_set)


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

    train_set, test_set = train_test_split(samples, 0.1)
    '''
    从实验结果来看，Bagging结合对率回归的效果较差
    结合决策树和朴素贝叶斯的效果较好，但朴素贝叶斯的提升效果不大
    对不减枝的决策树提升较大
    '''
    random_forest = Bagging(50, RandomDecisionTree())
    random_forest.train(train_set, attributes)
    s = random_forest.score(test_set)

    print('score =', s)

if __name__ == '__main__':
    test()
