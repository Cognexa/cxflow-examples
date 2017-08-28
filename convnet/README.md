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
cxflow dataset download convnet/config.yaml
cxflow train convnet/config.yaml
```

3. Train on GPU(s):
```
cxflow train convnet/config.yaml model.n_gpus:int=2
```

