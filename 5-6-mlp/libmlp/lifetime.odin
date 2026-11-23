package mlp

import "base:runtime"
@(export)
model_create :: proc "c" () -> rawptr {
	context = runtime.default_context()
	m := new([dynamic]Layer)
	return rawptr(m)
}

@(export)
model_add_layer :: proc "c" (handle: rawptr, in_feat, out_feat: u32, act: u8, init: u8) {
	context = runtime.default_context()
	m := (^[dynamic]Layer)(handle)
	append(m, layer_init(int(in_feat), int(out_feat), Activation(act), Init_Method(init)))
}

@(export)
model_destroy :: proc "c" (handle: rawptr) {
	context = runtime.default_context()
	m := (^[dynamic]Layer)(handle)
	for &l in m {
		delete(l.weights.data)
		delete(l.biases)
		delete(l.pre_act.data)
		delete(l.post_act.data)
		delete(l.weight_grad.data)
		delete(l.bias_grad)
	}
	delete(m^)
	free(m)
}
