import logging
import tarfile
import pickle
import os
import os.path as path
import shutil

import numpy as np
import cxflow as cx
import cv2


class CIFARDataset(cx.DownloadableDataset):
    """ Cifar100 dataset for image classification."""

    DOWNLOAD_URLS = ['https://github.com/Cognexa/cxflow-examples/releases/download/cifar100-dataset/cifar-100-python.tar.gz']
    FILENAME = 'cifar-100-python.tar.gz'

    FILE_STRUCTURE = {'train': {'images': b'data', 'labels': b'fine_labels'},
                    'test': {'images': b'data', 'labels': b'fine_labels'},
                    'meta': {'label_names': b'fine_label_names'}}

    def _configure_dataset(self, data_root=path.join('cifar100', '.cifar-data'), batch_size:int=100, label_class:int=-1, **kwargs) -> None:
        super()._configure_dataset(data_root=data_root, download_urls=CIFARDataset.DOWNLOAD_URLS)
        self._batch_size = batch_size
        self._label_class = label_class
        self._data = {'train': {}, 'test': {}, 'label_names': None}
        self._data_loaded = False

    def _load_data(self, normalize: bool=True) -> None:
        if not self._data_loaded:
            logging.info('Loading Cifar100 data to memory')

            file_path = path.join(self.data_root, CIFARDataset.FILENAME)
            if not path.exists(file_path):
                raise FileNotFoundError('File `{}` does not exist. '
                                        'Run `cxflow dataset download cifar100` first!'.format(file_path))

            with tarfile.open(file_path, 'r') as files:
                files.extractall(path=self.data_root)

            for key in CIFARDataset.FILE_STRUCTURE:
                with open(path.join(self.data_root, 'cifar-100-python', key), 'rb') as in_file:
                    dict = pickle.load(in_file, encoding='bytes')
                    for key_data in CIFARDataset.FILE_STRUCTURE[key]:
                        if key_data != 'label_names':
                            self._data[key][key_data] = dict[CIFARDataset.FILE_STRUCTURE[key][key_data]]
                        else:
                            self._data[key_data] = dict[CIFARDataset.FILE_STRUCTURE[key][key_data]]

            self._data['test']['images'] = self._data['test']['images'].reshape(self._data['test']['images'].shape[0], 3, 32, 32).transpose(0,2,3,1).astype("float32")
            self._data['train']['images'] = self._data['train']['images'].reshape(self._data['train']['images'].shape[0], 3, 32, 32).transpose(0,2,3,1).astype("float32")

            if normalize:
                mean = np.mean(self._data['train']['images'], axis=0, keepdims=True)
                std = np.std(self._data['train']['images'])

                self._data['train']['images'] = (self._data['train']['images'] - mean) / std
                self._data['test']['images'] = (self._data['test']['images'] - mean) / std

            self._data_loaded = True

    def visualize(self) -> None:
        self._load_data(normalize=False)

        if self._label_class >= 0:
            indexes = [i for i, label in enumerate(self._data['train']['labels']) if label == self._label_class][:100]
            i = -1
        else:
            indexes = list(range(100))

        output_array = np.hstack(tuple(self._data['train']['images'][indexes]))
        output_array = np.split(output_array, 10, axis=1)
        output_array = np.vstack(tuple(output_array))

        cv2.imwrite('cifar100_images.png', cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR))

    def train_stream(self) -> cx.Stream:
        self._load_data()

        permutation = np.random.permutation(len(self._data['train']['images']))
        self._data['train']['images'] = np.array(self._data['train']['images'])[permutation]
        self._data['train']['labels'] = np.array(self._data['train']['labels'])[permutation]


        for i in range(0, len(self._data['train']['images']), self._batch_size):

            aug_perm = np.random.rand(len(self._data['train']['images'][i: i + self._batch_size])) > 0.5

            for j, prob in enumerate(aug_perm):
                if prob:
                    self._data['train']['images'][i + j] = np.fliplr(self._data['train']['images'][i + j])

            yield {'images': self._data['train']['images'][i: i + self._batch_size],
                   'labels': self._data['train']['labels'][i: i + self._batch_size]}

    def test_stream(self) -> cx.Stream:
        self._load_data()
        for i in range(0, len(self._data['test']['images']), self._batch_size):
            yield {'images': self._data['test']['images'][i: i + self._batch_size],
                   'labels': self._data['test']['labels'][i: i + self._batch_size]}
