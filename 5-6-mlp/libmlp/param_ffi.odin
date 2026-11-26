package mlp

import "base:runtime"
@(export)
model_get_hyperparam :: proc "c" (handle: rawptr, param: u8) -> f32 {
	context = runtime.default_context()
	m := (^Model)(handle)
	switch Model_Hyperparam(param) {
	case .L1:          return m.l1
	case .L2:          return m.l2
	case .Grad_Clip:   return m.grad_clip
	case .Weight_Clip: return m.weight_clip
	case:              return 0.0
	}
}

@(export)
model_set_hyperparam :: proc "c" (handle: rawptr, param: u8, val: f32) {
	context = runtime.default_context()
	m := (^Model)(handle)
	switch Model_Hyperparam(param) {
	case .L1:          m.l1 = val
	case .L2:          m.l2 = val
	case .Grad_Clip:   m.grad_clip = val
	case .Weight_Clip: m.weight_clip = val
	}
}

@(export)
model_get_layer_param :: proc "c" (handle: rawptr, layer_idx: u32, param: u8) -> f32 {
	context = runtime.default_context()
	m := (^Model)(handle)
	if int(layer_idx) >= len(m.layers) do return 0.0
	switch Layer_Param(param) {
	case .Dropout: return m.layers[layer_idx].dropout_rate
	case:          return 0.0
	}
}

@(export)
model_set_layer_param :: proc "c" (handle: rawptr, layer_idx: u32, param: u8, val: f32) {
	context = runtime.default_context()
	m := (^Model)(handle)
	if int(layer_idx) >= len(m.layers) do return
	switch Layer_Param(param) {
	case .Dropout: m.layers[layer_idx].dropout_rate = val
	}
}
