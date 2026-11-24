import numpy as np

from libmlp import MLP, Activation, DenseLayer, Loss, set_rng_seed

BAD_SEED = 142
GOOD_SEED = 420

if __name__ == "__main__":
    set_rng_seed(GOOD_SEED)

    model = MLP(
        Loss.CCE,
        DenseLayer(inp_features=2, out_features=8, activation=Activation.RELU),
        DenseLayer(inp_features=8, out_features=2, activation=Activation.SOFTMAX),
    )

    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.int32)

    print("Initial Predictions:")
    initial_preds = model.forward(X)
    print(np.round(initial_preds, 4))
    print(f"Initial Cross-Entropy Loss: {model.evaluate(X, y):.4f}\n")

    learning_rate = 0.2
    epochs = 500

    print("Training...")
    for epoch in range(1, epochs + 1):
        loss = model.train_step(X, y, lr=learning_rate)
        if epoch % 100 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f}")

    # 4. Final evaluation
    final_preds = model.forward(X)
    print("\nFinal Predictions (Probabilities):")
    print(np.round(final_preds, 4))

    predicted_classes = np.argmax(final_preds, axis=1)
    print(f"Predicted Classes: {predicted_classes.tolist()} (Expected: {y.tolist()})")
    print(f"Final Cross-Entropy Loss: {model.evaluate(X, y):.4f}")
