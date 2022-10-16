import pandas as pd


class VuaFormat:
    """VUA-format data"""

    def __init__(self):
        #self.training_file = 'trainData.csv'
        #self.training_file = 'extraTrainData.csv'
        #self.training_file = 'hasoc-train.csv'
        self.training_file = 'olid-train-small.csv'
        self.task = None
        self.name = "VUA_format"
        self.train_data = None
        self.test_data = None

    def __str__(self):
        return self.name + ", " + self.task

    def load(self, data_dir, test_file='olid-test.csv'):
        self.train_data = pd.read_csv(
            data_dir + self.training_file, delimiter=",")
        self.test_data = pd.read_csv(data_dir + test_file, delimiter=",")

    def train_instances(self):
        """returns training instances and labels for a given task

        :return: X_train, y_train
        """
        return self.train_data['text'], self.train_data['labels']

    def test_instances(self):
        return self.test_data['text'], self.test_data['labels']
