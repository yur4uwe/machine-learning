package mlp

import "core:encoding/endian"
import "core:os"
import "core:slice"

magic_bytes :: "Tsar MLP1"

save_model :: proc(m: Model, f: ^os.File) {
	os.write_string(f, magic_bytes)

	layers := u32(len(m))
	b: [4]byte
	endian.put_u32(b[:], .Little, layers)
	os.write_slice(f, b[:])

	for &layer in m {
		inp_features := u32(layer.weights.rows)
		endian.put_u32(b[:], .Little, inp_features)
		os.write_slice(f, b[:])

		out_features := u32(layer.weights.cols)
		endian.put_u32(b[:], .Little, out_features)
		os.write_slice(f, b[:])

		activation := u8(layer.activation)
		os.write_byte(f, activation)

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
	buf := make([]byte, len(magic_bytes))
	os.read_full(f, buf)

	if string(buf) != magic_bytes {
		return nil, Format_Error(0)
	}

	len_buf := buf[:4]
	os.read_full(f, len_buf)

	l_cnt, ok := endian.get_u32(len_buf, .Little)
	if !ok {
		return nil, Format_Error(1)
	}

	ls := make([]Layer, l_cnt)
	model = ls

	for i in 0 ..< l_cnt {
		// 9 should be allocated for magic bytes already
		// 4 bytes inp_features, 4 bytes out_features, 1 byte activation
		os.read_full(f, buf)

		inp_features, out_features: u32

		inp_features, ok = endian.get_u32(buf[0:4], .Little)
		if !ok {
			return nil, Format_Error(2)
		}

		out_features, ok = endian.get_u32(buf[4:8], .Little)
		if !ok {
			return nil, Format_Error(3)
		}

		activation := buf[8]

		model[i].weights.rows = int(inp_features)
		model[i].weights.cols = int(out_features)
		model[i].activation = Activation(activation)

		weights_len := int(inp_features * out_features) * size_of(f64)
		weights_buf := make([]byte, weights_len)
		read := os.read_full(f, weights_buf) or_return
		if read != weights_len {
			return nil, File_Too_Short(0)
		}

		model[i].weights.data = slice.reinterpret([]f64, weights_buf)

		biases_len := int(out_features) * size_of(f64)
		biases_buf := make([]byte, biases_len)
		read = os.read_full(f, biases_buf) or_return
		if read != biases_len {
			return nil, File_Too_Short(1)
		}

		model[i].biases = slice.reinterpret([]f64, biases_buf)
	}

	return model, nil
}
