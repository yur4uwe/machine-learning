package mlp

import "core:math"
import "core:math/rand"

// Must mutate undelying array because Matrix is only a slice descriptor
Layer :: struct {
	weights:     Matrix,
	biases:      []f64,
	result:      Matrix,
	activation:  Activation,
	weight_grad: Matrix,
	bias_grad:   []f64,

	// mask etc
}

he_init :: proc(in_feat: int, out_feat: int, act: Activation) -> Layer {
	he_limit := math.sqrt(6.0 / f64(in_feat))

	weights := make([]f64, in_feat * out_feat)
	for i in 0 ..< in_feat * out_feat {
		weights[i] = rand.float64_range(-he_limit, he_limit)
	}
	m := Matrix{weights, {row_cnt = in_feat, col_cnt = out_feat}}

	bias := make([]f64, out_feat)
	return Layer{weights = m, biases = bias, activation = act}
}

layer_forward :: proc(prev: Matrix, curr: ^Layer) -> Matrix {
	data_size := prev.shape.row_cnt * curr.weights.shape.col_cnt

	if len(curr.result.data) != data_size {
		delete(curr.result.data)
		curr.result.data = make([]f64, data_size)
		curr.result.shape = {prev.shape.row_cnt, curr.weights.shape.col_cnt}
	}

	samples_cnt := prev.shape.row_cnt
	neurons_cnt := curr.weights.shape.col_cnt
	in_feat_cnt := prev.shape.col_cnt

	for sample_idx in 0 ..< samples_cnt {
		for neuron_idx in 0 ..< neurons_cnt {
			agg := curr.biases[neuron_idx]
			for feat_idx in 0 ..< in_feat_cnt {
				first_el := prev.data[sample_idx * stride(prev) + feat_idx]
				sec_el := curr.weights.data[feat_idx * stride(curr.weights) + neuron_idx]
				agg += first_el * sec_el
			}

			curr.result.data[sample_idx * stride(curr.result) + neuron_idx] = agg
		}
	}

	// activation MUST be non-nil anyway
	return layer_activation(curr.activation, curr.result)
}
