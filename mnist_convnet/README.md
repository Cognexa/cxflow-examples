# Convnet in cxflow-tensorflow
This folder contains convolutional network example written in **cxflow-tensorflow**
commented in greater detail in our [cxflow-tensorflow tutorial](https://tensorflow.cxflow.org/tutorial.html).

It is required to have **python 3.5+** and **pip** available in your system.

1. Install **cxflow-tensorflow** and download the examples (if not done yet):
```
pip3 install cxflow cxflow-tensorflow --upgrade
git clone https://github.com/Cognexa/cxflow-examples.git
cd cxflow-examples
```

2. Download the data and train the network:
```
cxflow dataset download mnist_convnet
cxflow train mnist_convnet
```

3. Train on two GPUs (omit `model.n_gpus=2` in order to train on CPU):
```
cxflow train mnist_convnet model.n_gpus=2
```
