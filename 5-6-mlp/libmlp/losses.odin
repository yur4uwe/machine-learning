package mlp

import "core:math"

Loss :: enum {
	CCE,
	BCE,
	MSE,
}

cce_loss :: proc "contextless" (preds, targets: Matrix) -> f64 {
	eps := 1e-15
	n := f64(preds.rows)
	sum := 0.0

	for v, i in preds.data {
		target := targets.data[i]
		if target > 0.0 {
			prob := clamp(v, eps, 1.0)
			sum += target * math.ln(prob)
		}
	}

	return (-1.0 / n) * sum
}

cce_loss_deriv :: proc "contextless" (preds, targets, errors: Matrix) {
	eps := 1e-15
	for v, i in preds.data {
		prob := clamp(v, eps, 1.0)
		errors.data[i] = -targets.data[i] / prob
	}
}

bce_loss :: proc "contextless" (preds, targets: Matrix) -> f64 {
	eps := 1e-15
	ndiv1 := 1.0 / f64(preds.rows * preds.cols)

	sum := 0.0
	for v, i in preds.data {
		tv := targets.data[i]
		prob := clamp(v, eps, 1.0 - eps)
		sum += tv * math.ln(prob) + (1.0 - tv) * math.ln(1.0 - prob)
	}

	return -ndiv1 * sum
}

bce_loss_deriv :: proc "contextless" (preds, targets, errors: Matrix) {
	eps := 1e-15
	for v, i in preds.data {
		prob := clamp(v, eps, 1.0 - eps)
		errors.data[i] = (prob - targets.data[i]) / (prob * (1.0 - prob))
	}
}

mse_loss :: proc "contextless" (preds, targets: Matrix) -> f64 {
	ndiv1 := 1.0 / f64(preds.rows * preds.cols)

	sum := 0.0
	for v, i in preds.data {
		diff := v - targets.data[i]
		sum += diff * diff
	}

	return ndiv1 * sum
}

mse_loss_deriv :: proc "contextless" (preds, targets, errors: Matrix) {
	for v, i in preds.data {
		errors.data[i] = v - targets.data[i]
	}
}

compute_model_loss :: proc "contextless" (model: Model, preds, targets: Matrix) -> f64 {
    loss: f64
	switch model.loss {
	case .CCE:
		loss = cce_loss(preds, targets)
	case .MSE:
		loss = mse_loss(preds, targets)
	case .BCE:
		loss = bce_loss(preds, targets)
	case:
		loss = 0.0
	}

    if model.l1 <= 0.0 && model.l2 <= 0.0 {
        return loss
    }

    sum_l1, sum_l2 := 0.0, 0.0
    for l in model.layers {
        for w_v in l.weights.data {
            sum_l1 += math.abs(w_v)
            sum_l2 += w_v * w_v
        }
    }

    if model.l1 > 0.0 {
        loss += model.l1 * sum_l1
    }
    if model.l2 > 0.0 {
        loss += (model.l2 / 2) * sum_l2
    }

    return loss
}

compute_initial_error :: proc "contextless" (
	loss: Loss,
	out_act: Activation,
	out_z: Matrix,
	preds: Matrix,
	targets: Matrix,
	errors: Matrix,
) {
	switch {
	case loss == .CCE && out_act == .Softmax,
	     loss == .BCE && out_act == .Sigmoid,
	     loss == .MSE && out_act == .None:
		// Fused shortcut
		for i in 0 ..< len(errors.data) {
			errors.data[i] = preds.data[i] - targets.data[i]
		}
	case:
		// General fallback:
		switch loss {
		case .CCE:
			cce_loss_deriv(preds, targets, errors)
		case .BCE:
			bce_loss_deriv(preds, targets, errors)
		case .MSE:
			mse_loss_deriv(preds, targets, errors)
		}
		layer_activation_deriv(out_act, out_z, errors)
	}
}
