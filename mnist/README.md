# MNIST hand-written digits recognition
This is simple **cxflow-tensorflow** implementation of MLP
neural network for hand-written character recognition.

It is required to have **python 3.5+** and **pip** available in your system.

To train your network with **cxflow** run the following commands:
```
pip3 install cxflow cxflow-tensorflow --upgrade
git clone https://github.com/Cognexa/cxflow-examples.git
cd cxflow-examples
cxflow dataset download mnist/config.yaml
cxflow train mnist/config.yaml
```
