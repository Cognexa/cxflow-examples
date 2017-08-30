# MNIST hand-written digits recognition
This is simple **cxflow-tensorflow** implementation of MLP
neural network for hand-written character recognition.

It is required to have **python 3.5+** and **pip** available in your system.

1. Install **cxflow-tensorflow** and download the examples (if not done yet):
```
pip3 install cxflow cxflow-tensorflow --upgrade
git clone https://github.com/Cognexa/cxflow-examples.git
cd cxflow-examples
```

2. Download the data and run the training:
```
cxflow dataset download mnist/config.yaml
cxflow train mnist/config.yaml
```
