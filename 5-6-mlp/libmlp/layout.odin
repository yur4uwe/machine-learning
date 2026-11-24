package mlp

import "core:encoding/endian"
import "core:os"
import "core:slice"

magic_bytes :: "Tsar MLP1"

save_model :: proc(m: Model, f: ^os.File) {
	os.write_string(f, magic_bytes)

	layers := u32(len(m.layers))

	floats := [?]f32{m.l1, m.l2, m.grad_clip, m.weight_clip}
	model_hdr: [size_of(u32) + size_of(u8) + len(floats) * size_of(f32)]byte

	start, end := 0, size_of(u32)
	endian.put_u32(model_hdr[start:end], .Little, layers)

	start = end; end += size_of(u8)
	model_hdr[start] = u8(m.loss)

	for f_val in floats {
		start = end; end += size_of(f32)
		endian.put_f32(model_hdr[start:end], .Little, f_val)
	}

	os.write_slice(f, model_hdr[:])

	layer_hdr: [2 * size_of(u32) + 2 * size_of(u8) + size_of(f32)]byte
	for &layer in m.layers {
		inp_features := u32(layer.weights.rows)
		start, end = 0, size_of(u32)
		endian.put_u32(layer_hdr[start:end], .Little, inp_features)

		out_features := u32(layer.weights.cols)
		start = end; end += size_of(u32)
		endian.put_u32(layer_hdr[start:end], .Little, out_features)

		start = end; end += size_of(u8)
		layer_hdr[start] = u8(layer.activation)

		start = end; end += size_of(u8)
		layer_hdr[start] = u8(layer.init_method)

		start = end; end += size_of(f32)
		endian.put_f32(layer_hdr[start:end], .Little, layer.dropout_rate)

		os.write_slice(f, layer_hdr[:])

		os.write_slice(f, layer.weights.data)
		os.write_slice(f, layer.biases)
	}
}

Format_Error :: distinct int
File_Too_Short :: distinct int

Model_Load_Error :: union {
	Format_Error,
	File_Too_Short,
	os.Error,
}

load_model :: proc(f: ^os.File) -> (model: Model, err: Model_Load_Error) {
	model_header_size := len(magic_bytes) + size_of(u32) + size_of(u8) + 4 * size_of(f32)
	layer_header_size := 2 * size_of(u32) + 2 * size_of(u8) + size_of(f32)

	max_buffer_size := max(model_header_size, layer_header_size)
	buf := make([]byte, max_buffer_size)
	defer delete(buf)

	model_header_buf := buf[:model_header_size]
	read := os.read_full(f, model_header_buf) or_return
	if read != model_header_size {
		return model, File_Too_Short(0)
	}

	start, end := 0, len(magic_bytes)
	if string(model_header_buf[start:end]) != magic_bytes {
		return model, Format_Error(0)
	}

	start = end; end += size_of(u32)
	l_cnt, ok := endian.get_u32(model_header_buf[start:end], .Little)
	if !ok {
		return model, Format_Error(1)
	}

	start = end; end += size_of(u8)
	loss_byte := model_header_buf[start]

	ls := make([dynamic]Layer, l_cnt)
	model = Model {
		layers = ls,
		loss   = Loss(loss_byte),
	}

	floats := [?]^f32{
		&model.l1, &model.l2, &model.grad_clip, &model.weight_clip,
	}

	for ref_f in floats {
		start = end; end += size_of(f32)
		ref_f^, ok = endian.get_f32(model_header_buf[start:end], .Little)
		if !ok {
			return model, Format_Error(5)
		}
	}

	layer_header_buf := buf[:layer_header_size]
	for i in 0 ..< l_cnt {
		read = os.read_full(f, layer_header_buf) or_return
		if read != layer_header_size {
			return model, File_Too_Short(1)
		}

		start, end = 0, size_of(u32)
		inp_features: u32
		inp_features, ok = endian.get_u32(layer_header_buf[start:end], .Little)
		if !ok {
			return model, Format_Error(2)
		}

		start = end; end += size_of(u32)
		out_features: u32
		out_features, ok = endian.get_u32(layer_header_buf[start:end], .Little)
		if !ok {
			return model, Format_Error(3)
		}

		start = end; end += size_of(u8)
		act_byte := layer_header_buf[start]

		start = end; end += size_of(u8)
		init_byte := layer_header_buf[start]

		start = end; end += size_of(f32)
		dropout_rate: f32
		dropout_rate, ok = endian.get_f32(layer_header_buf[start:end], .Little)
		if !ok {
			return model, Format_Error(4)
		}

		model.layers[i].weights.rows = int(inp_features)
		model.layers[i].weights.cols = int(out_features)
		model.layers[i].activation = Activation(act_byte)
		model.layers[i].init_method = Init_Method(init_byte)
		model.layers[i].dropout_rate = dropout_rate

		weights_len := int(inp_features * out_features) * size_of(f32)
		weights_buf := make([]byte, weights_len)
		read = os.read_full(f, weights_buf) or_return
		if read != weights_len {
			delete(weights_buf)
			return model, File_Too_Short(2)
		}

		model.layers[i].weights.data = slice.reinterpret([]f32, weights_buf)

		biases_len := int(out_features) * size_of(f32)
		biases_buf := make([]byte, biases_len)
		read = os.read_full(f, biases_buf) or_return
		if read != biases_len {
			delete(biases_buf)
			return model, File_Too_Short(3)
		}

		model.layers[i].biases = slice.reinterpret([]f32, biases_buf)
	}

	return model, nil
}
