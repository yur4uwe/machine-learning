package main

import "core:fmt"
import "core:math/rand"
import "core:os"
import "mlp"

main :: proc() {
	// xor_dataset := mlp.Matrix {
	// 	data = []f64{0, 0, 0, 1, 1, 0, 1, 1},
	// 	rows = 4,
	// 	cols = 2,
	// }
	//
	// xor_targets := []int{0, 1, 1, 0}
	//
	// rand.reset(142)
	//
	// model := mlp.Model{mlp.he_init(2, 8, .ReLU), mlp.he_init(8, 2, .Softmax)}
	//
	// mlp.train(model, xor_dataset, xor_targets, training_steps = 500, learning_rate = 0.2)
	//
	// f, err := os.open("test.mlp1", flags = os.File_Flags{.Create, .Write})
	// if err != nil {
	// 	fmt.eprintln(err)
	// }
	//
	// mlp.save_model(model, f)

	f, err := os.open("test.mlp1")
	if err != nil {
		fmt.eprintln(err)
	}
	model, lerr := mlp.load_model(f)
	if lerr != nil {
		fmt.eprintln(lerr)
	}

	probs := mlp.model_forward(model, mlp.Matrix{data = {1, 1}, rows = 1, cols = 2})
	fmt.println("Probabilities:")
	fmt.println(probs.data)
}
