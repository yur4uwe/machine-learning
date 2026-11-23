package mlp

import "core:fmt"
import "core:math"

Model :: []Layer

model_forward :: proc(m: Model, input: Matrix) -> Matrix {
	prev := input

	for &l in m {
		prev = layer_forward(prev, &l)
	}

	return prev
}

cross_entropy_loss :: proc "contextless" (probs: Matrix, targets: []int) -> f64 {
	eps := 1e-15
	n := f64(probs.rows)
	sum := 0.0
	for i in 0 ..< probs.rows {
		idx := i * stride(probs) + targets[i]
		prob := clamp(probs.data[idx], eps, 1.0 - eps)
		sum += math.ln(prob)
	}
	return (-1 / n) * sum
}

model_backward :: proc(model: Model, input, probs: Matrix, targets: []int) {
	// On all layers: rows are sample counts

	sample_cnt := probs.rows
	class_cnt := probs.cols

	// Its worth to find the layer with biggest feature count to reserve memory
	// for 2 of those

	max_feats := -1
	for layer in model {
		max_feats = max(max_feats, layer.weights.rows, layer.weights.cols)
	}

	current_errors_prealloc := make([]f64, sample_cnt * max_feats)
	backprop_errors_prealloc := make([]f64, sample_cnt * max_feats)
	defer delete(current_errors_prealloc)
	defer delete(backprop_errors_prealloc)

	errors := Matrix {
		data = current_errors_prealloc[:probs.rows * probs.cols],
		rows = probs.rows,
		cols = probs.cols,
	}

	// initial cross entropy errors
	for sample in 0 ..< sample_cnt {
		for class in 0 ..< class_cnt {
			idx := sample * stride(errors) + class
			if class != targets[sample] {
				errors.data[idx] = probs.data[idx]
			} else {
				errors.data[idx] = probs.data[idx] - 1.0
			}
		}
	}

	for l := len(model) - 1; l >= 0; l -= 1 {
		curr_l := &model[l]
		prev_res: Matrix
		if l > 0 {
			prev_res = model[l - 1].result
		} else {
			prev_res = input
		}

		in_feat_cnt := prev_res.cols
		out_feat_cnt := curr_l.weights.cols

		grad_w := Matrix {
			data = make([]f64, in_feat_cnt * out_feat_cnt),
			rows = in_feat_cnt,
			cols = out_feat_cnt,
		}

		// Weight gradient calculation X^T * delta
		// X is (Sample, Feature) and X^T is (Feature, Sample)
		// delta is (Sample, Class)
		matmul(prev_res, errors, dst = grad_w, scale = 1.0 / f64(sample_cnt), transpose_lhs = true)

		grad_b := make([]f64, out_feat_cnt)

		// Bias gradient calculation: B * delta
		// B is (Class)
		// delta is (Sample, Class)
		for class in 0 ..< out_feat_cnt {
			sum := 0.0
			for sample in 0 ..< sample_cnt {
				sum += errors.data[sample * stride(errors) + class]
			}
			grad_b[class] = sum / f64(sample_cnt)
		}

		curr_l.weight_grad = grad_w
		curr_l.bias_grad = grad_b

		if l == 0 {
			continue
		}

		prev_errors := Matrix {
			// I would like to reuse space for errors from previous steps
			data = backprop_errors_prealloc[:sample_cnt * in_feat_cnt],
			rows = sample_cnt,
			cols = in_feat_cnt,
		}

		// Backprop error: delta * W^T
		// W is (Features, Class) and W^T is (Class, Features)
		// delta is still (Sample, Class)
		matmul(errors, curr_l.weights, dst = prev_errors, transpose_rhs = true)

		prev_activation := model[l - 1].activation
		for i in 0 ..< len(prev_errors.data) {
			prev_errors.data[i] *= layer_activation_deriv(prev_activation, prev_res.data[i])
		}

		errors = prev_errors

		// errors hold backprop_errors_prealloc
		// but we need it to reference current_errors_prealloc
		// so we swap them
		backprop_errors_prealloc, current_errors_prealloc =
			current_errors_prealloc, backprop_errors_prealloc
	}
}

model_optimizer :: proc(m: Model, learning_rate: f64) {
	for &l in m {
		for &w, i in l.weights.data {
			w -= learning_rate * l.weight_grad.data[i]
		}

		for &v, i in l.biases {
			v -= learning_rate * l.bias_grad[i]
		}
	}
}

train :: proc(
	m: Model,
	input: Matrix,
	targets: []int,
	training_steps: int = 500,
	learning_rate: f64 = 0.01,
) {
	loss: f64
	for step in 1 ..= training_steps {
		probs := model_forward(m, input)

		loss = cross_entropy_loss(probs, targets)
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
					fmt.printf("%.2f", probs.data[i * stride(probs) + j])
				}
				fmt.print("]")
			}
			fmt.println("]")
		}

		model_backward(m, input, probs, targets)
		model_optimizer(m, learning_rate)
	}

	fmt.printfln("Final loss: %f", loss)
}
