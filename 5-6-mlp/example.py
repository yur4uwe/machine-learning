import numpy as np
from libmlp import CCE, MLP, Activation, DenseLayer

if __name__ == "__main__":
    # 1. Define model architecture
    model = MLP(
        DenseLayer(inp_features=2, out_features=8, activation=Activation.RELU),
        DenseLayer(inp_features=8, out_features=2, activation=Activation.SOFTMAX),
    )

    # 2. XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([0, 1, 1, 0], dtype=np.int32)

    print("Initial Predictions:")
    initial_preds = model.forward(X)
    print(np.round(initial_preds, 4))
    print(f"Initial Cross-Entropy Loss: {CCE(initial_preds, y):.4f}\n")

    # 3. Training loop
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
    print(f"Final Cross-Entropy Loss: {CCE(final_preds, y):.4f}")
