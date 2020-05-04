from sklearn.model_selection import GroupKFold as gkf
from sklearn.model_selection import StratifiedKFold as skf


class mySplitter:
    def __init__(self, split_type, split_num, shuffle, random_state, logger):
        self.logger = logger
        self.split_type = split_type
        self.split_num = split_num
        self.shuffle = shuffle
        self.random_state = random_state

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
        else:
            raise NotImplementedError(f'split_type: {self.split_type}')
        return fold
