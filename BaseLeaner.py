from common import *


class BaseLeaner:
    @classmethod
    def new(cls):
        instance = cls.__new__(cls)
        instance.__init__()
        return instance

    def train(self, sample_set: SampleSet, attributes: list):
        pass

    def predict(self, sample: Sample):
        pass

    def score(self, sample_set):
        pass
