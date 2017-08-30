# Cxflow Examples

This is the repository containing various examples for [cxflow](Cognexa/cxflow).
Before you dive in, make sure you have already read the [official cxflow documentation](https://cxflow.org/)
and the 10-minutes tutorials for both [cxflow](https://cxflow.org/tutorial.html) and
[cxflow-tensorflow](https://tensorflow.cxflow.org/tutorial.html). 

The repository is organized into multiple directories.
Firstly, there is a `datasets/` directory which contains the implementation of various datasets that are used in the following examples.

Secondly, example directories are provided.
We suggest you to walk them through in the following order, however, it is completely up to you.
Each directory contains a `Readme` file which describes in detail the process of obtaining the data, training a simple model and in some cases additional information.

- [majority/](majority/)` - full implementation of the [official cxflow tutorial](https://cxflow.org/tutorial.html).
- [mnist_mlp/](mnist_mlp/) - a simple experiment with MNIST dataset and shallow multi-layer perceptron.
- [mnist_convnet`](mnist_convnet/) - a more advanced experiment with MNIST dataset and a convolutional network.
- [imdb/](imdb/)` - an implementation of sentiment analysis using the IMDB dataset and a bi-directional reccurent neural network.

## Contribution
In case you have created a nice experiment using **cxflow**, do not hesitate to fork this repository and make a pull request.

## License
The source codes within this repository are distributed under [MIT License](LICENSE).   
