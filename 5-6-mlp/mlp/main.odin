package main

import "mlp"

main :: proc() {
	xor_dataset := mlp.Matrix {
		data  = []f64{0, 0, 0, 1, 1, 0, 1, 1},
		shape = {4, 2},
	}

	xor_targets := []int{0, 1, 1, 0}

	model := mlp.Model{mlp.he_init(2, 8, .ReLU), mlp.he_init(8, 2, .Softmax)}

	mlp.train(model, xor_dataset, xor_targets, training_steps = 500, learning_rate = 0.2)
}
