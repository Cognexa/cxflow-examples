import tensorflow.contrib.keras as K
import numpy as np
from cxflow.datasets import BaseDataset, AbstractDataset


class IMDBDataset(BaseDataset):
    """
    IMDB dataset for review binary sentiment classification.

    Reference: <http://ai.stanford.edu/~amaas/data/sentiment/>
    """

    def _init_with_kwargs(self, batch_size:int=100, maxlen:int=400, **kwargs) -> None:
        self._batch_size = batch_size
        self._maxlen = maxlen
        self._data_loaded = False

    def _load_data(self) -> None:
        if not self._data_loaded:
            (self._train_x, self._train_y), \
            (self._test_x, self._test_y) = K.datasets.imdb.load_data(maxlen=self._maxlen+1)
            self._train_x = K.preprocessing.sequence.pad_sequences(self._train_x)
            self._test_x = K.preprocessing.sequence.pad_sequences(self._test_x)

            self._data_loaded = True

    @property
    def vocab_size(self) -> int:
        self._load_data()
        return max(np.max(self._train_x), np.max(self._test_x), np.max(self._train_y), np.max(self._test_y))

    @property
    def maxlen(self) -> int:
        return self._maxlen

    def train_stream(self) -> AbstractDataset.Stream:
        self._load_data()
        perm = np.random.permutation(len(self._train_x))
        perm_x = self._train_x[perm]
        perm_y = self._train_y[perm]
        for i in range(0, len(self._train_x), self._batch_size):
            yield {'x': perm_x[i: i + self._batch_size],
                   'y': perm_y[i: i + self._batch_size]}

    def test_stream(self) -> AbstractDataset.Stream:
        self._load_data()
        for i in range(0, len(self._test_x), self._batch_size):
            yield {'x': self._test_x[i: i + self._batch_size],
                   'y': self._test_y[i: i + self._batch_size]}
