package mlp

import "core:math"
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
	for &x in m.data {
		x = max(0.0, x)
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

dsigmoid :: proc "contextless" (z: Matrix, grad: Matrix) {
	for i in 0 ..< len(grad.data) {
		s := 1.0 / (1.0 + math.exp(-z.data[i]))
		grad.data[i] *= s * (1.0 - s)
	}
}

tanh_act :: proc "contextless" (m: Matrix) {
	for &x in m.data {
		x = math.tanh(x)
	}
}

dtanh :: proc "contextless" (z: Matrix, grad: Matrix) {
	for i in 0 ..< len(grad.data) {
		t := math.tanh(z.data[i])
		grad.data[i] *= (1.0 - t * t)
	}
}

leaky_relu :: proc "contextless" (m: Matrix, alpha: f64 = 0.01) {
	for &x in m.data {
		if x <= 0.0 {
			x *= alpha
		}
	}
}

dleaky_relu :: proc "contextless" (z: Matrix, grad: Matrix, alpha: f64 = 0.01) {
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
		sum := 0.0
		for &val in row {
			val = math.exp(val - maximum)
			sum += val
		}

		for &val in row {
			val /= sum
		}
	}
}

dsoftmax :: proc "contextless" (z: Matrix, grad: Matrix) {
	// Vector-Jacobian product for general Softmax backward
	for r in 0 ..< z.rows {
		row_start := r * stride(z)
		z_row := z.data[row_start : row_start + z.cols]
		g_row := grad.data[row_start : row_start + grad.cols]

		maximum := slice.max(z_row)
		sum := 0.0
		for val in z_row {
			sum += math.exp(val - maximum)
		}

		dot := 0.0
		for c in 0 ..< z.cols {
			p := math.exp(z_row[c] - maximum) / sum
			dot += g_row[c] * p
		}

		for c in 0 ..< z.cols {
			p := math.exp(z_row[c] - maximum) / sum
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

layer_activation_deriv :: proc "contextless" (typ: Activation, z: Matrix, grad: Matrix) {
	switch typ {
	case .ReLU:
		drelu(z, grad)
	case .Sigmoid:
		dsigmoid(z, grad)
	case .Tanh:
		dtanh(z, grad)
	case .LeakyReLU:
		dleaky_relu(z, grad)
	case .Softmax:
		dsoftmax(z, grad)
	case .None:
		fallthrough
	case:
	}
}
