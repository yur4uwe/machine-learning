package mlp

import "base:runtime"
import "core:math/rand"
import "core:mem"

@(export)
set_seed :: proc "c" (seed: u64) {
	context = runtime.default_context()
	rand.reset(seed)
}

// -----------------------------------------------------------------------------
// Execution & Training
// -----------------------------------------------------------------------------

@(export)
model_forward_ffi :: proc "c" (
	handle: rawptr,
	x_data: [^]f32,
	rows, cols: u32,
	out_probs: [^]f32,
) {
	context = runtime.default_context()
	m := (^Model)(handle)
	input := Matrix{data = x_data[:rows * cols], rows = int(rows), cols = int(cols)}
	probs := model_forward(m^, input)
	mem.copy(out_probs, raw_data(probs.data), len(probs.data) * size_of(f32))
}

@(export)
model_evaluate :: proc "c" (
	handle: rawptr,
	x_data: [^]f32,
	rows, cols: u32,
	y_data: [^]f32,
) -> f32 {
	context = runtime.default_context()
	m := (^Model)(handle)
	out_dim := m.layers[len(m.layers) - 1].weights.cols
	input := Matrix{data = x_data[:rows * cols], rows = int(rows), cols = int(cols)}
	targets := Matrix{data = y_data[:int(rows) * out_dim], rows = int(rows), cols = out_dim}

	old_training := m.is_training
	m.is_training = false
	defer m.is_training = old_training

	probs := model_forward(m^, input)
	return compute_model_loss(m^, probs, targets)
}

@(export)
model_train_step :: proc "c" (
	handle: rawptr,
	x_data: [^]f32,
	rows, cols: u32,
	y_data: [^]f32,
	lr: f32,
) -> f32 {
	context = runtime.default_context()
	m := (^Model)(handle)
	out_dim := m.layers[len(m.layers) - 1].weights.cols
	input := Matrix{data = x_data[:rows * cols], rows = int(rows), cols = int(cols)}
	targets := Matrix{data = y_data[:int(rows) * out_dim], rows = int(rows), cols = out_dim}

	old_training := m.is_training
	m.is_training = true
	defer m.is_training = old_training

	probs := model_forward(m^, input)
	loss := compute_model_loss(m^, probs, targets)
	model_backward(m, input, probs, targets)
	model_optimizer(m^, lr)
	return loss
}

@(export)
model_set_training :: proc "c" (handle: rawptr, training: bool) {
	context = runtime.default_context()
	m := (^Model)(handle)
	m.is_training = training
}

@(export)
model_is_training :: proc "c" (handle: rawptr) -> bool {
	context = runtime.default_context()
	m := (^Model)(handle)
	return m.is_training
}

// -----------------------------------------------------------------------------
// Diagnostics
// -----------------------------------------------------------------------------

@(export)
model_get_dead_neurons :: proc "c" (
	handle: rawptr,
	x_data: [^]f32,
	rows, cols: u32,
	layer_idx: i32,
) -> f32 {
	context = runtime.default_context()
	m := (^Model)(handle)
	input := Matrix{data = x_data[:rows * cols], rows = int(rows), cols = int(cols)}
	return model_dead_neurons(m^, input, layer_idx)
}

// -----------------------------------------------------------------------------
// Architecture & Layer Data
// -----------------------------------------------------------------------------

@(export)
model_num_layers :: proc "c" (handle: rawptr) -> u32 {
	context = runtime.default_context()
	m := (^Model)(handle)
	return u32(len(m.layers))
}

@(export)
model_output_dim :: proc "c" (handle: rawptr) -> u32 {
	context = runtime.default_context()
	m := (^Model)(handle)
	if len(m.layers) == 0 do return 0
	return u32(m.layers[len(m.layers) - 1].weights.cols)
}

@(export)
model_layer_dims :: proc "c" (handle: rawptr, layer_idx: u32, in_feat, out_feat: ^u32) {
	context = runtime.default_context()
	m := (^Model)(handle)
	in_feat^ = u32(m.layers[layer_idx].weights.rows)
	out_feat^ = u32(m.layers[layer_idx].weights.cols)
}

@(export)
model_get_layer_params :: proc "c" (handle: rawptr, layer_idx: u32, dst_w: [^]f32, dst_b: [^]f32) {
	context = runtime.default_context()
	m := (^Model)(handle)
	w := m.layers[layer_idx].weights.data
	b := m.layers[layer_idx].biases
	mem.copy(dst_w, raw_data(w), len(w) * size_of(f32))
	mem.copy(dst_b, raw_data(b), len(b) * size_of(f32))
}

@(export)
model_set_layer_params :: proc "c" (handle: rawptr, layer_idx: u32, src_w: [^]f32, src_b: [^]f32) {
	context = runtime.default_context()
	m := (^Model)(handle)
	if int(layer_idx) >= len(m.layers) do return
	w := m.layers[layer_idx].weights.data
	b := m.layers[layer_idx].biases
	if len(w) > 0 && src_w != nil do mem.copy(raw_data(w), src_w, len(w) * size_of(f32))
	if len(b) > 0 && src_b != nil do mem.copy(raw_data(b), src_b, len(b) * size_of(f32))
}

@(export)
model_get_layer_gradients :: proc "c" (handle: rawptr, layer_idx: u32, dst_w: [^]f32, dst_b: [^]f32) {
	context = runtime.default_context()
	m := (^Model)(handle)
	if int(layer_idx) >= len(m.layers) do return
	w_grad := m.layers[layer_idx].weight_grad.data
	b_grad := m.layers[layer_idx].bias_grad
	if len(w_grad) > 0 && dst_w != nil do mem.copy(dst_w, raw_data(w_grad), len(w_grad) * size_of(f32))
	if len(b_grad) > 0 && dst_b != nil do mem.copy(dst_b, raw_data(b_grad), len(b_grad) * size_of(f32))
}
