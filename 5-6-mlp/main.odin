package main

import "core:fmt"
import "core:math/rand"
import mlp "libmlp"

main :: proc() {
	xor_dataset := mlp.Matrix {
		data = []f64{0, 0, 0, 1, 1, 0, 1, 1},
		rows = 4,
		cols = 2,
	}

	xor_targets := []i32{0, 1, 1, 0}

	rand.reset(420)

	model := mlp.Model{mlp.layer_init(2, 8, .ReLU), mlp.layer_init(8, 2, .Softmax)}

	learning_rate := 0.4
	training_steps := 500
	loss: f64
	for step in 1 ..= training_steps {
		probs := mlp.model_forward(model, xor_dataset)

		loss = mlp.cross_entropy_loss(probs, xor_targets)
		if step % 100 == 0 {
			fmt.printfln("step %d: loss: %f", step, loss)
			fmt.print("Predictions:\n[")
			for i in 0 ..< probs.rows {
				if i > 0 {
					fmt.print(", ")
				}
				fmt.print("[")
				for j in 0 ..< probs.cols {
					if j > 0 {
						fmt.print(", ")
					}
					fmt.printf("%.2f", probs.data[i * mlp.stride(probs) + j])
				}
				fmt.print("]")
			}
			fmt.println("]")
		}

		mlp.model_backward(model, xor_dataset, probs, xor_targets)
		mlp.model_optimizer(model, learning_rate)
	}

	fmt.printfln("Final loss: %f", loss)

	// f, err := os.open("test.mlp1", flags = os.File_Flags{.Create, .Write})
	// if err != nil {
	// 	fmt.eprintln(err)
	// }

	// mlp.save_model(model, f)
	//
	// f, err := os.open("test.mlp1")
	// if err != nil {
	// 	fmt.eprintln(err)
	// }
	// model, lerr := mlp.load_model(f)
	// if lerr != nil {
	// 	fmt.eprintln(lerr)
	// }
	//
	// probs := mlp.model_forward(model, mlp.Matrix{data = {1, 1}, rows = 1, cols = 2})
	// fmt.println("Probabilities:")
	// fmt.println(probs.data)
}
