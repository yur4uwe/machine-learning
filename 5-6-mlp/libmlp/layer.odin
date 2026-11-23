package mlp

import "core:math"
import "core:math/rand"

// Must mutate undelying array because Matrix is only a slice descriptor
Layer :: struct {
	weights:     Matrix,
	biases:      []f64,
	pre_act:     Matrix,
	post_act:    Matrix,
	activation:  Activation,
	weight_grad: Matrix,
	bias_grad:   []f64,
}

Init_Method :: enum {
	Auto,
	Zeros,
	Uniform,
	Kaiming,
	Xavier,
}

layer_init :: proc(
	in_feat, out_feat: int,
	act: Activation,
	ini_type: Init_Method = .Auto,
) -> Layer {
	weights := make([]f64, in_feat * out_feat)

	limit: f64

	switch ini_type {
	case .Auto:
		switch act {
		case .ReLU, .LeakyReLU:
			limit = math.sqrt(6.0 / f64(in_feat))
		case .Softmax, .None, .Sigmoid, .Tanh:
			limit = math.sqrt(6.0 / f64(in_feat + out_feat))
		}
	case .Kaiming:
		limit = math.sqrt(6.0 / f64(in_feat))
	case .Xavier:
		limit = math.sqrt(6.0 / f64(in_feat + out_feat))
	case .Uniform:
		limit = 1 / math.sqrt(f64(in_feat))
	case .Zeros:
	// Odin by default zeroes the memory out
	}

	if ini_type != .Zeros {
		for i in 0 ..< in_feat * out_feat {
			weights[i] = rand.float64_range(-limit, limit)
		}
	}

	w := Matrix{weights, in_feat, out_feat}
	b := make([]f64, out_feat)

	return Layer{weights = w, biases = b, activation = act}
}

he_init :: proc(in_feat: int, out_feat: int, act: Activation) -> Layer {
	he_limit := math.sqrt(6.0 / f64(in_feat))

	weights := make([]f64, in_feat * out_feat)
	for i in 0 ..< in_feat * out_feat {
		weights[i] = rand.float64_range(-he_limit, he_limit)
	}
	m := Matrix{weights, in_feat, out_feat}

	bias := make([]f64, out_feat)
	return Layer{weights = m, biases = bias, activation = act}
}

layer_forward :: proc(prev: Matrix, curr: ^Layer) -> Matrix {
	data_size := prev.rows * curr.weights.cols

	if len(curr.pre_act.data) != data_size {
		delete(curr.pre_act.data)
		curr.pre_act.data = make([]f64, data_size)
		curr.pre_act.rows = prev.rows
		curr.pre_act.cols = curr.weights.cols
	}

	matmul(prev, curr.weights, curr.biases, curr.pre_act)

	if len(curr.post_act.data) != data_size {
		delete(curr.post_act.data)
		curr.post_act = mat_full_copy(curr.pre_act)
	} else {
		copy(curr.post_act.data, curr.pre_act.data)
	}

	layer_activation(curr.activation, curr.post_act)

	return curr.post_act
}
