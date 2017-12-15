import logging
import tarfile
import pickle
import os
import os.path as path

import numpy as np
import cxflow as cx

DOWNLOAD_URL = ['https://github.com/Cognexa/cxflow-examples/releases/download/cifar100-dataset/cifar-100-python.tar.gz']
FILENAME = 'cifar-100-python.tar.gz'

DICT_STRUCTURE = {'train_images': b'data',
                  'train_labels_fine': b'fine_labels',
                  'train_labels_coarse': b'coarse_labels',
                  'test_images': b'data',
                  'test_labels_fine': b'fine_labels',
                  'test_labels_coarse': b'coarse_labels'}


class CIFARDataset(cx.DownloadableDataset):
    """ Cifar100 dataset for image classification."""

    def _configure_dataset(self, data_root=path.join('cifar100', '.cifar-data'), batch_size:int=100, **kwargs) -> None:
        self._batch_size = batch_size
        self._data_root = data_root
        self._download_urls = DOWNLOAD_URL
        self._data = {}
        self._data_loaded = False

    def _load_data(self) -> None:
        if not self._data_loaded:
            logging.info('Loading Cifar100 data to memory')

            file_path = path.join(self.data_root, FILENAME)
            if not path.exists(file_path):
                raise FileNotFoundError('File `{}` does not exist. '
                                        'Run `cxflow dataset download <path-to-config>` first!'.format(file_path))

            with tarfile.open(file_path, 'r:gz') as files:
                files.extractall(self.data_root)

            for key in DICT_STRUCTURE:
                if 'train' in key:
                    with open(path.join(self.data_root, 'cifar-100-python', 'train'), 'rb') as train_file:
                        dict = pickle.load(train_file, encoding='bytes')
                        self._data[key] = dict[DICT_STRUCTURE[key]]
                else:
                    with open(path.join(self.data_root, 'cifar-100-python', 'test'), 'rb') as test_file:
                        dict = pickle.load(test_file, encoding='bytes')
                        self._data[key] = dict[DICT_STRUCTURE[key]]
            self._data_loaded = True

    def train_stream(self) -> cx.Stream:
        self._load_data()
        for i in range(0, len(self._data['train_labels']), self._batch_size):
            yield {'images': self._data['train_images'][i: i + self._batch_size],
                   'fine_labels': self._data['train_labels_fine'][i: i + self._batch_size],
                   'coarse_labels': self._data['train_labels_coarse'][i: i + self._batch_size]}

    def test_stream(self) -> cx.Stream:
        self._load_data()
        for i in range(0, len(self._data['test_labels']), self._batch_size):
            yield {'images': self._data['test_images'][i: i + self._batch_size],
                   'fine_labels': self._data['test_labels_fine'][i: i + self._batch_size],
                   'coarse_labels': self._data['test_labels_coarse'][i: i + self._batch_size]}
