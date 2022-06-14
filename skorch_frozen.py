import numpy as np
import skorch
from sklearn.base import TransformerMixin
from skorch import NeuralNet
from skorch.utils import to_numpy


class NeuralNetTransformer(NeuralNet, TransformerMixin):
    '''
    https://github.com/skorch-dev/skorch/issues/482
    '''

    def get_loss(self, y_pred, y_true, X, **kwargs):
        y_pred, _ = y_pred
        return super().get_loss(y_pred, y_true=X, X=X, **kwargs)

    def transform(self, X):
        out = []
        for outs in self.forward_iter(X, training=False):
            outs = outs[1] if isinstance(outs, tuple) else outs
            out.append(to_numpy(outs))
        transforms = np.concatenate(out, 0)
        return transforms


class FrozenNeuralNetTransformer(NeuralNetTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialize()

    def fit(self, X, y=None):
        # does nothing
        return self


class FrozenNeuralNetClassifier(skorch.NeuralNetClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialize()

    def fit(self, X, y=None):
        # does nothing
        return self


if __name__ == '__main__':
    import torch
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import FunctionTransformer

    layer = torch.nn.Linear(20, 10)
    for p in layer.parameters():
        p.requires_grad = False
    frozen_weights = layer.weight.data.numpy()
    transformer = FrozenNeuralNetTransformer(layer, torch.nn.MSELoss)
    pipeline = make_pipeline(
        FunctionTransformer(func=np.float32, inverse_func=np.float64),
        transformer,
        LogisticRegression(),
    )
    X = np.random.randn(100, 20)
    y = np.random.randint(0, 2, 100)
    pipeline.fit(X, y)
    print(pipeline.predict(X))
    print(pipeline)
    assert np.all(pipeline[1].module.weight.data.numpy() == frozen_weights)
