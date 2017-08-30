# IMDB review sentiment dataset
In this example, we train a simple GRU recurrent network for sentiment analysis for movie reviews.

The network is trained on IMDB dataset <http://ai.stanford.edu/~amaas/data/sentiment/>.

Trained networ can be easily applied to a new review as explained bellow.

It is required to have **python 3.5+** and **pip** available in your system.

1. Install **cxflow-tensorflow** and download the examples (if not done yet):
```
pip3 install cxflow cxflow-tensorflow --upgrade
git clone https://github.com/Cognexa/cxflow-examples.git
cd cxflow-examples
```

2. Train the bi-directional GRU network
```
cxflow train imdb
```

You should reach test accuracy around 89.5%, the best model will be saved in `log/IMDB_GRU_<dir_name>`.

3. Obtain predictions interactively:
```
cxflow predict log/<your training directory>
```

After a moment, you should see the following prompt:
```
Type the review or leave empty to end: I was not sure at the beginning, but in the end it was a good movie. The performance of Emma Stone was outstanding.
Predicted sentiment: positive

Type the review or leave empty to end: I like Tarantino films, but this one was simply awful. The dialogues did not make any sense and the camera was terrible and shaky.
Predicted sentiment: negative

Type the review or leave empty to end:
```

Can you fool your network?
