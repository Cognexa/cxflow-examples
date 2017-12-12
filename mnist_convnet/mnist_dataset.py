import logging
import gzip
import struct
import urllib.request
import os
import os.path as path
from typing import Iterable

import numpy as np
import cxflow as cx

DOWNLOAD_ROOT = 'https://github.com/Cognexa/cxflow-examples/releases/download/mnist-dataset/'
FILENAMES = {'train_images': 'train-images-idx3-ubyte.gz',
             'train_labels': 'train-labels-idx1-ubyte.gz',
             'test_images': 't10k-images-idx3-ubyte.gz',
             'test_labels': 't10k-labels-idx1-ubyte.gz'}


class MNISTDataset(cx.DownloadableDataset):
    """ MNIST dataset for hand-written digits recognition."""

    def _configure_dataset(self, data_root=path.join('mnist_convnet', '.mnist-data'), batch_size:int=100, **kwargs) -> None:
        self._batch_size = batch_size
        self._data_root = data_root
        self._download_urls = [path.join(DOWNLOAD_ROOT, filename) for filename in FILENAMES.values()]
        self._data = {}
        self._data_loaded = False

    @property
    def data_root(self) -> str:
        return super().data_root

    @property
    def download_urls(self) -> Iterable[str]:
        return super().download_urls

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

    def train_stream(self) -> cx.Stream:
        self._load_data()
        for i in range(0, len(self._data['train_labels']), self._batch_size):
            yield {'images': self._data['train_images'][i: i + self._batch_size],
                   'labels': self._data['train_labels'][i: i + self._batch_size]}

    def test_stream(self) -> cx.Stream:
        self._load_data()
        for i in range(0, len(self._data['test_labels']), self._batch_size):
            yield {'images': self._data['test_images'][i: i + self._batch_size],
                   'labels': self._data['test_labels'][i: i + self._batch_size]}
