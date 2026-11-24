package mlp

import "base:runtime"
@(export)
model_create :: proc "c" (loss: u8) -> rawptr {
	context = runtime.default_context()
	m := new(Model)
	m.loss = Loss(loss)
	return rawptr(m)
}

@(export)
model_add_layer :: proc "c" (handle: rawptr, in_feat, out_feat: u32, act: u8, init: u8, dropout_rate: f32) {
	context = runtime.default_context()
	m := (^Model)(handle)
	l := layer_init(int(in_feat), int(out_feat), Activation(act), Init_Method(init))
	l.dropout_rate = dropout_rate
	append(&m.layers, l)
}

@(export)
model_destroy :: proc "c" (handle: rawptr) {
	context = runtime.default_context()
	m := (^Model)(handle)
	for &l in m.layers {
		delete(l.weights.data)
		delete(l.biases)
		delete(l.pre_act.data)
		delete(l.post_act.data)
		delete(l.weight_grad.data)
		delete(l.bias_grad)
        delete(l.dropout_mask.data)
	}
	delete(m.err_buf_a)
	delete(m.err_buf_b)
	delete(m.layers)
	free(m)
}
