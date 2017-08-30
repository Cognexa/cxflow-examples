# Majority example
This is the very first toy example for **cxflow** framework commented in greater detail in our [tutorial](https://cxflow.org/tutorial.html).

It is required to have **python 3.5+** and **pip** available in your system.

1. Install **cxflow-tensorflow** and download the examples (if not done yet):
```
pip3 install cxflow cxflow-tensorflow --upgrade
git clone https://github.com/Cognexa/cxflow-examples.git
cd cxflow-examples
```

2. Train the majority network
```
cxflow train majority
```

The best network will be saved in `log/MajorityExample_<dir_name>`.

3. Resume the training
```
cxflow resume log/MajorityExample_<dir_name>
```

The very first epoch should perform similarly well to the last epoch of the previous training.
