# CIFAR-100 Dataset
In this example, we train a wide residual network for image classification task.

We use CIFAR-100 dataset <https://www.cs.toronto.edu/~kriz/cifar.html>. In `cifar_dataset.py` we show how to use your own dataset. You can check how to download your data, prepare it and then use for training your model.

The network architecture was proposed in this paper  <https://arxiv.org/abs/1605.07146>. We decided to use architecture which has depth 28 and widening factor is 10. According to the paper, this architecture should has test error around 20.50% on CIFAR-100 dataset.  

It is required to have **python 3.5+** and **pip** available in your system.

1. Install **cxflow-tensorflow** and download the examples (if not done yet):
```
pip3 install cxflow cxflow-tensorflow --upgrade
git clone https://github.com/Cognexa/cxflow-examples.git
cd cxflow-examples
```

2. Download the data and train the network:
```
cxflow dataset download cifar100
cxflow train cifar100
```
The best network will be saved in `log/WideResnet_<dir_name>`.

3. Resume the training
```
cxflow resume cifar100
```
