package mlp

Model :: struct {
	layers: [dynamic]Layer,
	loss:   Loss,
}

model_forward :: proc(m: Model, input: Matrix) -> Matrix {
	prev := input

	for &l in m.layers {
		prev = layer_forward(prev, &l)
	}

	return prev
}


model_backward :: proc(model: Model, input, probs: Matrix, targets: Matrix) {
	// On all layers: rows are sample counts

	sample_cnt := probs.rows
	class_cnt := probs.cols

	// Its worth to find the layer with biggest feature count to reserve memory
	// for 2 of those

	max_feats := -1
	for layer in model.layers {
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

	last_l := &model.layers[len(model.layers) - 1]
	compute_initial_error(model.loss, last_l.activation, last_l.pre_act, probs, targets, errors)

	for l := len(model.layers) - 1; l >= 0; l -= 1 {
		curr_l := &model.layers[l]
		prev_res: Matrix
		if l > 0 {
			prev_res = model.layers[l - 1].post_act
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

		prev_activation := model.layers[l - 1].activation
		prev_z := model.layers[l - 1].pre_act
		layer_activation_deriv(prev_activation, prev_z, prev_errors)

		errors = prev_errors

		// errors hold backprop_errors_prealloc
		// but we need it to reference current_errors_prealloc
		// so we swap them
		backprop_errors_prealloc, current_errors_prealloc =
			current_errors_prealloc, backprop_errors_prealloc
	}
}

model_optimizer :: proc(m: Model, learning_rate: f64) {
	for &l in m.layers {
		for &w, i in l.weights.data {
			w -= learning_rate * l.weight_grad.data[i]
		}

		for &v, i in l.biases {
			v -= learning_rate * l.bias_grad[i]
		}
	}
}
