from cxflow.hooks import AbstractHook


class IMDBPredict(AbstractHook):
    SENTIMENTS = {0: 'negative', 1: 'positive'}

    def after_batch(self, stream_name, batch_data):
        print('Predicted sentiment: {}'.format(IMDBPredict.SENTIMENTS[batch_data['predictions'][0]]))
        print()
