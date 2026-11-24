package mlp

import "core:math"
Model :: struct {
	layers: [dynamic]Layer,
	loss:   Loss,

    // Hyperparams where 0.0 == disabled
    l1:          f64,
    l2:          f64,
    grad_clip:   f64,
    weight_clip: f64,

    is_training: bool
}

model_forward :: proc(m: Model, input: Matrix) -> Matrix {
	prev := input

	for &l in m.layers {
        if m.is_training {
            prev = layer_forward_train(prev, &l)
        } else {
            prev = layer_forward(prev, &l)
        }
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

        grad_w: Matrix
        grad_b: []f64
        if len(curr_l.weight_grad.data) != in_feat_cnt * out_feat_cnt {
            grad_w = Matrix {
                data = make([]f64, in_feat_cnt * out_feat_cnt),
                rows = in_feat_cnt,
                cols = out_feat_cnt,
            }
        } else {
            // Overwrite the new data
            grad_w = curr_l.weight_grad
        }
        if len(curr_l.bias_grad) != out_feat_cnt {
            grad_b = make([]f64, out_feat_cnt)
        } else {
            grad_b = curr_l.bias_grad
        }

		// Weight gradient calculation X^T * delta
		// X is (Sample, Feature) and X^T is (Feature, Sample)
		// delta is (Sample, Class)
		matmul(prev_res, errors, dst = grad_w, scale = 1.0 / f64(sample_cnt), transpose_lhs = true)


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

        // Mask out unactivated neurons so they don't start contributing to the error
        prev_l_dropout_rate := model.layers[l - 1].dropout_rate
        prev_l_dropout_mask := model.layers[l - 1].dropout_mask
        if prev_l_dropout_rate > 0.0 && model.is_training && len(prev_l_dropout_mask.data) > 0 {
            for &err, i in prev_errors.data {
                err *= prev_l_dropout_mask.data[i]
            }
        }

		prev_activation := model.layers[l - 1].activation
		prev_z := model.layers[l - 1].pre_act
		layer_activation_deriv(prev_activation, prev_z, prev_errors)

		errors = prev_errors

		// errors hold backprop_errors_prealloc
		// but we need it to reference current_errors_prealloc
		// so we swap fat pointers to them
		backprop_errors_prealloc, current_errors_prealloc =
			current_errors_prealloc, backprop_errors_prealloc
	}
}

model_optimizer :: proc(m: Model, learning_rate: f64) {
    scale := 1.0
    uses_reg := m.l1 > 0.0 || m.l2 > 0.0
    uses_grad_clipping := m.grad_clip > 0.0

    if uses_reg || uses_grad_clipping {
        norm_sq := 0.0
        for &l in m.layers {
            for &w, i in l.weights.data {
                if m.l1 > 0.0 do l.weight_grad.data[i] += m.l1 * math.sign(w)
                if m.l2 > 0.0 do l.weight_grad.data[i] += m.l2 * w
                if uses_grad_clipping do norm_sq += l.weight_grad.data[i] * l.weight_grad.data[i]
            }

            if !uses_grad_clipping {
                continue
            }
            for g in l.bias_grad {
                norm_sq += g * g
            }
        }

        if uses_grad_clipping {
            global_norm := math.sqrt(norm_sq)
            if global_norm > m.grad_clip {
                scale = m.grad_clip / global_norm
            }
        }
    }

    eff_lr := learning_rate * scale

    for &l in m.layers {
        for &w, i in l.weights.data {
            w -= eff_lr * l.weight_grad.data[i]
            if m.weight_clip > 0.0 {
                w = clamp(w, -m.weight_clip, m.weight_clip)
            }
        }

        for &b, i in l.biases {
            b -= eff_lr * l.bias_grad[i]
        }
    }
}

Model_Hyperparam :: enum u8 {
	L1          = 0,
	L2          = 1,
	Grad_Clip   = 2,
	Weight_Clip = 3,
}

Layer_Param :: enum u8 {
	Dropout = 0,
}

model_dead_neurons :: proc(m: Model, input: Matrix, layer_idx: i32) -> f64 {
	if len(m.layers) == 0 do return 0.0

	count_dead_in_layer :: proc(l: ^Layer, sample_cnt: int) -> (int, int) {
		neuron_cnt := l.weights.cols
		dead_cnt := 0
		st := stride(l.pre_act)

		for j in 0 ..< neuron_cnt {
			is_dead := true
			for s in 0 ..< sample_cnt {
				if l.pre_act.data[s * st + j] > 0.0 {
					is_dead = false
					break
				}
			}
			if is_dead {
				dead_cnt += 1
			}
		}

		return dead_cnt, neuron_cnt
	}

	num_layers := len(m.layers)
	num_hidden := max(0, num_layers - 1)

	inference_m := m
	inference_m.is_training = false
	model_forward(inference_m, input)

	if layer_idx >= 0 {
		idx := int(layer_idx)
		if idx >= num_layers do return 0.0
		dead, total := count_dead_in_layer(&m.layers[idx], input.rows)
		if total == 0 do return 0.0
		return f64(dead) / f64(total) * 100.0
	} else {
		if num_hidden == 0 do return 0.0
		total_dead := 0
		total_neurons := 0
		for l in 0 ..< num_hidden {
			dead, total := count_dead_in_layer(&m.layers[l], input.rows)
			total_dead += dead
			total_neurons += total
		}
		if total_neurons == 0 do return 0.0
		return f64(total_dead) / f64(total_neurons) * 100.0
	}
}
