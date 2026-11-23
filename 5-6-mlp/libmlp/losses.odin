package mlp

import "core:math"

Losses :: enum {
	CCE,
	BCE,
	MSE,
}

cross_entropy_loss :: proc "contextless" (probs: Matrix, targets: []i32) -> f64 {
	eps := 1e-15
	n := f64(probs.rows)
	sum := 0.0
	for i in 0 ..< probs.rows {
		idx := i * stride(probs) + int(targets[i])
		prob := clamp(probs.data[idx], eps, 1.0 - eps)
		sum += math.ln(prob)
	}
	return (-1 / n) * sum
}
