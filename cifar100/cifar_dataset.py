import logging
import tarfile
import pickle
import os
import os.path as path
import shutil

import numpy as np
import cxflow as cx
from matplotlib import pyplot as plt

DOWNLOAD_URL = ['https://github.com/Cognexa/cxflow-examples/releases/download/cifar100-dataset/cifar-100-python.tar.gz']
FILENAME = 'cifar-100-python.tar.gz'

DOWNLOAD_STRUCTURE = {'train': {'images': b'data', 'labels': b'fine_labels'},
                  'test': {'images': b'data', 'labels': b'fine_labels'},
                  'meta': {'label_names': b'fine_label_names'}}


class CIFARDataset(cx.DownloadableDataset):
    """ Cifar100 dataset for image classification."""

    def _configure_dataset(self, data_root=path.join('cifar100', '.cifar-data'), batch_size:int=100, **kwargs) -> None:
        self._batch_size = batch_size
        self._data_root = data_root
        self._download_urls = DOWNLOAD_URL
        self._data = {'train': {}, 'test': {}, 'label_names': None}
        self._data_loaded = False

    def _load_data(self) -> None:
        if not self._data_loaded:
            logging.info('Loading Cifar100 data to memory')

            file_path = path.join(self.data_root, FILENAME)
            if not path.exists(file_path):
                raise FileNotFoundError('File `{}` does not exist. '
                                        'Run `cxflow dataset download <path-to-config>` first!'.format(file_path))

            with tarfile.open(file_path, 'r') as files:
                files.extractall(path=self.data_root)

            for key in DOWNLOAD_STRUCTURE:
                with open(path.join(self.data_root, 'cifar-100-python', key), 'rb') as in_file:
                    dict = pickle.load(in_file, encoding='bytes')
                    for key_data in DOWNLOAD_STRUCTURE[key]:
                        if key_data != 'label_names':
                            self._data[key][key_data] = dict[DOWNLOAD_STRUCTURE[key][key_data]]
                        else:
                            self._data[key_data] = dict[DOWNLOAD_STRUCTURE[key][key_data]]

            self._data['train']['images'] = self._data['train']['images'].reshape(self._data['train']['images'].shape[0], 3, 32, 32).transpose(0,2,3,1).astype("uint8")
            self._data['test']['images'] = self._data['test']['images'].reshape(self._data['test']['images'].shape[0], 3, 32, 32).transpose(0,2,3,1).astype("uint8")

            shutil.rmtree(path.join(self.data_root, 'cifar-100-python'))

            self._data_loaded = True

    def grid_of_images(self, one_class=False, label_class=0) -> None:
        self._load_data()
        # one_class = True

        if one_class:
            indexes = [i for i, label in enumerate(self._data['train']['labels']) if label == label_class][:100]
            i = -1

        for col in range(10):
            for row in range(10):

                if one_class:
                    i += 1
                    index = indexes[i]
                else:
                    index = row + (col * 10)

                if row != 0:
                    row_array = np.concatenate((row_array, self._data['train']['images'][index]), axis=1)
                else:
                    row_array = self._data['train']['images'][index]

            if col != 0:
                output_array = np.concatenate((output_array, row_array), axis=0)
            else:
                output_array = row_array
            row_array = None

        plt.imshow(output_array)
        plt.axis('off')
        plt.show()

    def train_stream(self) -> cx.Stream:
        self._load_data()
        for i in range(0, len(self._data['train']['images']), self._batch_size):
            yield {'images': self._data['train']['images'][i: i + self._batch_size],
                   'labels': self._data['train']['labels'][i: i + self._batch_size]}

    def test_stream(self) -> cx.Stream:
        self._load_data()
        for i in range(0, len(self._data['test']['images']), self._batch_size):
            yield {'images': self._data['test']['images'][i: i + self._batch_size],
                   'labels': self._data['test']['labels'][i: i + self._batch_size]}
