package mlp

import "core:math"
import "core:simd"
import "core:slice"

Activation :: enum {
	None,
	ReLU,
	Softmax,
	Sigmoid,
	Tanh,
	LeakyReLU,
}

relu :: proc "contextless" (m: Matrix) {
	zero := V8(0.0)
	n := len(m.data)
	i := 0
	for ; i + 8 <= n; i += 8 {
		store_v8(m.data, i, simd.max(load_v8(m.data, i), zero))
	}
	for ; i < n; i += 1 {
		m.data[i] = max(0.0, m.data[i])
	}
}

drelu :: proc "contextless" (z: Matrix, grad: Matrix) {
	for i in 0 ..< len(grad.data) {
		if z.data[i] <= 0.0 {
			grad.data[i] = 0.0
		}
	}
}

sigmoid :: proc "contextless" (m: Matrix) {
	for &x in m.data {
		x = 1.0 / (1.0 + math.exp(-x))
	}
}

dsigmoid :: proc "contextless" (a: Matrix, grad: Matrix, act_scale: f32 = 1.0) {
	one := V8(1.0)
	sv := V8(act_scale)
	n := len(grad.data)
	i := 0
	for ; i + 8 <= n; i += 8 {
		s := load_v8(a.data, i) * sv
		g := load_v8(grad.data, i)
		store_v8(grad.data, i, g * s * (one - s))
	}
	for ; i < n; i += 1 {
		s := a.data[i] * act_scale
		grad.data[i] *= s * (1.0 - s)
	}
}

tanh_act :: proc "contextless" (m: Matrix) {
	for &x in m.data {
		x = math.tanh(x)
	}
}

dtanh :: proc "contextless" (a: Matrix, grad: Matrix, act_scale: f32 = 1.0) {
	one := V8(1.0)
	sv := V8(act_scale)
	n := len(grad.data)
	i := 0
	for ; i + 8 <= n; i += 8 {
		t := load_v8(a.data, i) * sv
		g := load_v8(grad.data, i)
		store_v8(grad.data, i, g * (one - t * t))
	}
	for ; i < n; i += 1 {
		t := a.data[i] * act_scale
		grad.data[i] *= (1.0 - t * t)
	}
}

leaky_relu :: proc "contextless" (m: Matrix, alpha: f32 = 0.01) {
	for &x in m.data {
		if x <= 0.0 {
			x *= alpha
		}
	}
}

dleaky_relu :: proc "contextless" (z: Matrix, grad: Matrix, alpha: f32 = 0.01) {
	for i in 0 ..< len(grad.data) {
		if z.data[i] <= 0.0 {
			grad.data[i] *= alpha
		}
	}
}

softmax :: proc "contextless" (m: Matrix) {
	for i in 0 ..< m.rows {
		// use a numeric trick where:
		// exp(xi - C) / sum(exp(xj - C)) == exp(xi) / sum(exp(xj))

		row_start := i * stride(m)
		row := m.data[row_start:row_start + m.cols]

		maximum := slice.max(row)
		sum: f32 = 0.0
		for &val in row {
			val = math.exp(val - maximum)
			sum += val
		}

		inv_sum := 1.0 / sum
		for &val in row {
			val *= inv_sum
		}
	}
}

dsoftmax :: proc "contextless" (a: Matrix, grad: Matrix, act_scale: f32 = 1.0) {
	// Vector-Jacobian product for general Softmax backward using cached post_act
	for r in 0 ..< a.rows {
		row_start := r * stride(a)
		a_row := a.data[row_start : row_start + a.cols]
		g_row := grad.data[row_start : row_start + grad.cols]

		dot: f32 = 0.0
		for c in 0 ..< a.cols {
			p := a_row[c] * act_scale
			dot += g_row[c] * p
		}

		for c in 0 ..< a.cols {
			p := a_row[c] * act_scale
			g_row[c] = p * (g_row[c] - dot)
		}
	}
}

layer_activation :: proc "contextless" (typ: Activation, m: Matrix) {
	switch typ {
	case .ReLU:
		relu(m)
	case .Softmax:
		softmax(m)
	case .Sigmoid:
		sigmoid(m)
	case .Tanh:
		tanh_act(m)
	case .LeakyReLU:
		leaky_relu(m)
	case .None:
		fallthrough
	case:
	}
}

layer_activation_deriv :: proc "contextless" (
	typ: Activation,
	z: Matrix,
	a: Matrix,
	grad: Matrix,
	act_scale: f32 = 1.0,
) {
	switch typ {
	case .ReLU:
		drelu(z, grad)
	case .Sigmoid:
		dsigmoid(a, grad, act_scale)
	case .Tanh:
		dtanh(a, grad, act_scale)
	case .LeakyReLU:
		dleaky_relu(z, grad)
	case .Softmax:
		dsoftmax(a, grad, act_scale)
	case .None:
		fallthrough
	case:
	}
}
