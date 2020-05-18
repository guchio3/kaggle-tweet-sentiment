import pandas as pd
from sklearn.model_selection import GroupKFold as gkf
from sklearn.model_selection import StratifiedKFold as skf


class mySplitter:
    def __init__(self, split_type, split_num, shuffle,
                 random_state, abhishek5, abhishek8, logger):
        self.logger = logger
        self.split_type = split_type
        self.split_num = split_num
        self.shuffle = shuffle
        self.random_state = random_state
        self.abhishek5 = abhishek5
        self.abhishek8 = abhishek8

    def split(self, x, y, group=None):
        if self.split_type == 'skf':
            if group is not None:
                self.logger.warn(
                    'the group is set for skf, '
                    'which is not used.'
                )
            fold = skf(
                self.split_num,
                shuffle=self.shuffle,
                random_state=self.random_state
            ).split(x, y)
        elif self.split_type == 'gkf':
            fold = gkf(self.split_num).split(x, y, group)
        elif self.split_type == 'abhishek5':
            fold = []
            fold_df = pd.read_csv(self.abhishek5)
            for i in range(5):
                fold.append((
                            fold_df.query('kfold == {i}').index.tolist(),
                            fold_df.query('kfold != {i}').index.tolist(),
                            ))
        elif self.split_type == 'abhishek8':
            fold = []
            fold_df = pd.read_csv(self.abhishek8)
            for i in range(8):
                fold.append((
                            fold_df.query('kfold == {i}').index.tolist(),
                            fold_df.query('kfold != {i}').index.tolist(),
                            ))
        else:
            raise NotImplementedError(f'split_type: {self.split_type}')
        return fold
