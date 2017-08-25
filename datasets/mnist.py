import logging
import gzip
import struct
import urllib.request
import os.path as path
import numpy as np
from cxflow.datasets import BaseDataset, AbstractDataset

DOWNLOAD_ROOT = 'https://github.com/Cognexa/cxflow-examples/releases/download/mnist-dataset/'
FILENAMES = {'train_images': 'train-images-idx3-ubyte.gz',
             'train_labels': 'train-labels-idx1-ubyte.gz',
             'test_images': 't10k-images-idx3-ubyte.gz',
             'test_labels': 't10k-labels-idx1-ubyte.gz'}


class MNISTDataset(BaseDataset):
    """ MNIST dataset for hand-written digits recognition."""

    def _init_with_kwargs(self, data_root='mnist', batch_size:int=100, **kwargs) -> None:
        self._batch_size = batch_size
        self._data_root = data_root
        self._data = {}
        self._data_loaded = False

    def _load_data(self) -> None:
        if not self._data_loaded:
            logging.info('Loading MNIST data to memory')
            for key in FILENAMES:
                file_path = path.join(self._data_root, FILENAMES[key])
                if not path.exists(file_path):
                    raise FileNotFoundError('File `{}` does not exist. '
                                            'Run `cxflow dataset download <path-to-config>` first!'.format(file_path))
                with gzip.open(file_path, 'rb') as file:
                    if 'images' in key:
                        _, _, rows, cols = struct.unpack(">IIII", file.read(16))
                        self._data[key] = np.frombuffer(file.read(), dtype=np.uint8).reshape(-1, rows, cols)
                    else:
                        _ = struct.unpack(">II", file.read(8))
                        self._data[key] = np.frombuffer(file.read(), dtype=np.int8)
            self._data_loaded = True

    def train_stream(self) -> AbstractDataset.Stream:
        self._load_data()
        for i in range(0, len(self._data['train_labels']), self._batch_size):
            yield {'images': self._data['train_images'][i: i + self._batch_size],
                   'labels': self._data['train_labels'][i: i + self._batch_size]}

    def test_stream(self) -> AbstractDataset.Stream:
        self._load_data()
        for i in range(0, len(self._data['test_labels']), self._batch_size):
            yield {'images': self._data['test_images'][i: i + self._batch_size],
                   'labels': self._data['test_labels'][i: i + self._batch_size]}

    def download(self) -> None:
        """Download method may be invoked with `cxflow dataset download <path-to-config>`."""
        for part in FILENAMES.values():
            target = path.join(self._data_root, part)
            if path.exists(target):
                logging.info('\t%s already exists', target)
            else:
                logging.info('\tdownloading %s', target)
                urllib.request.urlretrieve(DOWNLOAD_ROOT+part, target)
