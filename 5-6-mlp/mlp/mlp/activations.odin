package mlp

import "core:math"
import "core:slice"

Activation :: enum {
	None,
	ReLU,
	Softmax,
}

relu :: proc(m: Matrix) -> Matrix {
	for &x in m.data {
		x = max(0.0, x)
	}
	return m
}

drelu :: proc(v: f64) -> f64 {
	return v > 0.0 ? 1.0 : 0.0
}

softmax :: proc(m: Matrix) -> Matrix {
	for i in 0 ..< m.shape.row_cnt {
		// use a numeric trick where:
		// exp(xi - C) / sum(exp(xj - C)) == exp(xi) / sum(exp(xj))

		row_start := i * stride(m)
		row := m.data[row_start:row_start + m.shape.col_cnt]

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
	return m
}

layer_activation :: proc(typ: Activation, m: Matrix) -> Matrix {
	switch typ {
	case .ReLU:
		return relu(m)
	case .Softmax:
		return softmax(m)
	case .None:
		fallthrough
	case:
		return m
	}
}

layer_activation_deriv :: proc(typ: Activation, v: f64) -> f64 {
	switch typ {
	case .ReLU:
		return drelu(v)
	case .Softmax:
		fallthrough
	case .None:
		fallthrough
	case:
		return 1.0
	}
}
