import numpy.random as npr

from cxflow.datasets import BaseDataset, AbstractDataset


class MajorityDataset(BaseDataset):
    """
    Toy dataset for the majority task.

    See ../majority/README.md for basic usage. or <https://cognexa.github.io/cxflow/tutorial> for detailed tutorial.
    """

    def _init_with_kwargs(self, n_examples: int, dim: int, batch_size: int, **kwargs) -> None:
        """
        Randomly generate the data and split them to train and test streams in 4:1 ratio.

        :param n_examples: number of examples to be generated
        :param dim: example dimension
        :param batch_size: batch size
        :param kwargs: additional unused ``**kwargs``
        """
        self.batch_size = batch_size

        x = npr.random_integers(0, 1, n_examples * dim).reshape(n_examples, dim)
        y = x.sum(axis=1) > int(dim/2)

        self._train_x, self._train_y = x[:int(.8 * n_examples)], y[:int(.8 * n_examples)]
        self._test_x, self._test_y = x[int(.8 * n_examples):], y[int(.8 * n_examples):]

    def train_stream(self) -> AbstractDataset.Stream:
        for i in range(0, len(self._train_x), self.batch_size):
            yield {'x': self._train_x[i: i + self.batch_size],
                   'y': self._train_y[i: i + self.batch_size]}

    def test_stream(self) -> AbstractDataset.Stream:
        for i in range(0, len(self._test_x), self.batch_size):
            yield {'x': self._test_x[i: i + self.batch_size],
                   'y': self._test_y[i: i + self.batch_size]}
