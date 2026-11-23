package mlp

import "base:runtime"
import "core:math/rand"
import "core:mem"

@(export)
set_seed :: proc "c" (seed: u64) {
	context = runtime.default_context()
	rand.reset(seed)
}

@(export)
model_forward_ffi :: proc "c" (
	handle: rawptr,
	x_data: [^]f64,
	rows, cols: u32,
	out_probs: [^]f64, // NumPy pre-allocated buffer
) {
	context = runtime.default_context()
	m := (^Model)(handle)
	input := Matrix {
		data = x_data[:rows * cols],
		rows = int(rows),
		cols = int(cols),
	}

	probs := model_forward(m^, input)
	mem.copy(out_probs, raw_data(probs.data), len(probs.data) * size_of(f64))
}

@(export)
model_output_dim :: proc "c" (handle: rawptr) -> u32 {
	context = runtime.default_context()
	m := (^Model)(handle)
	if len(m.layers) == 0 do return 0
	return u32(m.layers[len(m.layers) - 1].weights.cols)
}

@(export)
model_evaluate :: proc "c" (
	handle: rawptr,
	x_data: [^]f64,
	rows, cols: u32,
	y_data: [^]f64,
) -> f64 {
	context = runtime.default_context()
	m := (^Model)(handle)
	input := Matrix {
		data = x_data[:rows * cols],
		rows = int(rows),
		cols = int(cols),
	}
	out_dim := m.layers[len(m.layers) - 1].weights.cols
	targets := Matrix {
		data = y_data[:int(rows) * out_dim],
		rows = int(rows),
		cols = out_dim,
	}

	probs := model_forward(m^, input)
	return compute_model_loss(m.loss, probs, targets)
}

@(export)
model_train_step :: proc "c" (
	handle: rawptr,
	x_data: [^]f64,
	rows, cols: u32,
	y_data: [^]f64,
	lr: f64,
) -> f64 {
	context = runtime.default_context()
	m := (^Model)(handle)
	input := Matrix {
		data = x_data[:rows * cols],
		rows = int(rows),
		cols = int(cols),
	}
	out_dim := m.layers[len(m.layers) - 1].weights.cols
	targets := Matrix {
		data = y_data[:int(rows) * out_dim],
		rows = int(rows),
		cols = out_dim,
	}

	probs := model_forward(m^, input)
	loss := compute_model_loss(m.loss, probs, targets)
	model_backward(m^, input, probs, targets)
	model_optimizer(m^, lr)
	return loss
}

@(export)
model_num_layers :: proc "c" (handle: rawptr) -> u32 {
	context = runtime.default_context()
	m := (^Model)(handle)
	return u32(len(m.layers))
}

@(export)
model_layer_dims :: proc "c" (handle: rawptr, layer_idx: u32, in_feat, out_feat: ^u32) {
	context = runtime.default_context()
	m := (^Model)(handle)
	in_feat^ = u32(m.layers[layer_idx].weights.rows)
	out_feat^ = u32(m.layers[layer_idx].weights.cols)
}

@(export)
model_get_layer_params :: proc "c" (handle: rawptr, layer_idx: u32, dst_w: [^]f64, dst_b: [^]f64) {
	context = runtime.default_context()
	m := (^Model)(handle)
	w := m.layers[layer_idx].weights.data
	b := m.layers[layer_idx].biases
	mem.copy(dst_w, raw_data(w), len(w) * size_of(f64))
	mem.copy(dst_b, raw_data(b), len(b) * size_of(f64))
}

@(export)
model_get_weights :: proc "c" (handle: rawptr, layer_idx: u32, dst: [^]f64) {
	context = runtime.default_context()
	m := (^Model)(handle)
	w := m.layers[layer_idx].weights.data
	mem.copy(dst, raw_data(w), len(w) * size_of(f64))
}
