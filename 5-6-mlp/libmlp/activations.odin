package mlp

import "core:math"
import "core:slice"

Activation :: enum {
	None,
	ReLU,
	Softmax,
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

layer_activation :: proc "contextless" (typ: Activation, m: Matrix) {
	switch typ {
	case .ReLU:
		relu(m)
	case .Softmax:
		softmax(m)
	case .None:
		fallthrough
	case:
	}
}

layer_activation_deriv :: proc "contextless" (typ: Activation, z: Matrix, grad: Matrix) {
	switch typ {
	case .ReLU:
		drelu(z, grad)
	case .Softmax, .None:
		fallthrough
	case:
	}
}
