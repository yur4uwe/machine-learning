import numpy as np

from libmlp import MLP

model = MLP.load("test.mlp1")

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([0, 1, 1, 0], dtype=np.int32)

preds = model.forward(X)
print(np.round(preds, 4))
print(f"Loaded model loss: {model.evaluate(X, y):.4f}\n")
