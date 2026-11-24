import ctypes
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .signatures import lib


class Activation(IntEnum):
    NONE = 0
    RELU = 1
    SOFTMAX = 2
    SIGMOID = 3
    TANH = 4
    LEAKY_RELU = 5


class Loss(IntEnum):
    CCE = 0
    BCE = 1
    MSE = 2


class LayerInit(IntEnum):
    AUTO = 0
    ZEROS = 1
    UNIFORM = 2
    KAIMING = 3
    XAVIER = 4


class Hyperparam(IntEnum):
    L1 = 0
    L2 = 1
    GRAD_CLIP = 2
    WEIGHT_CLIP = 3


class LayerParam(IntEnum):
    DROPOUT = 0


def set_rng_seed(seed: int):
    lib.set_seed(seed)


@dataclass
class DenseLayer:
    inp_features: int
    out_features: int
    activation: Activation = Activation.RELU
    init_method: LayerInit = LayerInit.AUTO
    dropout_rate: float = 0.0


class MLP:
    def __init__(
        self,
        loss: Loss = Loss.CCE,
        *layers: DenseLayer,
        l1: float = 0.0,
        l2: float = 0.0,
        grad_clip: float = 0.0,
        weight_clip: float = 0.0,
    ):
        self._handle = lib.model_create(int(loss))
        if l1 > 0.0:
            self.l1 = l1
        if l2 > 0.0:
            self.l2 = l2
        if grad_clip > 0.0:
            self.grad_clip = grad_clip
        if weight_clip > 0.0:
            self.weight_clip = weight_clip

        for layer in layers:
            lib.model_add_layer(
                self._handle,
                layer.inp_features,
                layer.out_features,
                layer.activation,
                layer.init_method,
                layer.dropout_rate,
            )

    @property
    def l1(self) -> float:
        return lib.model_get_hyperparam(self._handle, Hyperparam.L1)

    @l1.setter
    def l1(self, value: float):
        lib.model_set_hyperparam(self._handle, Hyperparam.L1, float(value))

    @property
    def l2(self) -> float:
        return lib.model_get_hyperparam(self._handle, Hyperparam.L2)

    @l2.setter
    def l2(self, value: float):
        lib.model_set_hyperparam(self._handle, Hyperparam.L2, float(value))

    @property
    def grad_clip(self) -> float:
        return lib.model_get_hyperparam(self._handle, Hyperparam.GRAD_CLIP)

    @grad_clip.setter
    def grad_clip(self, value: float):
        lib.model_set_hyperparam(self._handle, Hyperparam.GRAD_CLIP, float(value))

    @property
    def weight_clip(self) -> float:
        return lib.model_get_hyperparam(self._handle, Hyperparam.WEIGHT_CLIP)

    @weight_clip.setter
    def weight_clip(self, value: float):
        lib.model_set_hyperparam(self._handle, Hyperparam.WEIGHT_CLIP, float(value))

    def get_layer_dropout(self, layer_idx: int) -> float:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"Layer index {layer_idx} out of range [0, {self.num_layers})")
        return lib.model_get_layer_param(self._handle, layer_idx, LayerParam.DROPOUT)

    def set_layer_dropout(self, layer_idx: int, dropout_rate: float):
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"Layer index {layer_idx} out of range [0, {self.num_layers})")
        lib.model_set_layer_param(self._handle, layer_idx, LayerParam.DROPOUT, float(dropout_rate))

    @classmethod
    def load(cls, path: str) -> "MLP":
        handle = lib.model_load_params(path.encode("utf-8"))
        if not handle:
            raise OSError(f"failed to load model from {path}")

        self = cls.__new__(cls)
        self._handle = handle
        return self

    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float) -> float:
        # X and y memory is pinned by NumPy; Odin reads directly via pointer
        X = np.ascontiguousarray(X, dtype=np.float64)
        if y.ndim == 1:
            if self._out_dim > 1:
                y = np.eye(self._out_dim, dtype=np.float64)[y]
            else:
                y = y.reshape(-1, 1).astype(np.float64)
        else:
            y = np.ascontiguousarray(y, dtype=np.float64)

        X_c = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        y_c = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        return lib.model_train_step(self._handle, X_c, X.shape[0], X.shape[1], y_c, lr)

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = np.empty((X.shape[0], self._out_dim), dtype=np.float64)
        X = np.ascontiguousarray(X, dtype=np.float64)
        X_c = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        out_c = out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        lib.model_forward_ffi(self._handle, X_c, X.shape[0], X.shape[1], out_c)
        return out

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        X = np.ascontiguousarray(X, dtype=np.float64)
        if y.ndim == 1:
            if self._out_dim > 1:
                y = np.eye(self._out_dim, dtype=np.float64)[y]
            else:
                y = y.reshape(-1, 1).astype(np.float64)
        else:
            y = np.ascontiguousarray(y, dtype=np.float64)

        X_c = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        y_c = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        return lib.model_evaluate(self._handle, X_c, X.shape[0], X.shape[1], y_c)

    @property
    def num_layers(self) -> int:
        return lib.model_num_layers(self._handle)

    def get_layer_parameters(self, layer_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"Layer index {layer_idx} out of range [0, {self.num_layers})")

        in_feat = ctypes.c_uint32()
        out_feat = ctypes.c_uint32()
        lib.model_layer_dims(
            self._handle,
            layer_idx,
            ctypes.byref(in_feat),
            ctypes.byref(out_feat),
        )

        w = np.empty((in_feat.value, out_feat.value), dtype=np.float64)
        b = np.empty((out_feat.value,), dtype=np.float64)

        w_c = w.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        b_c = b.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        lib.model_get_layer_params(self._handle, layer_idx, w_c, b_c)
        return w, b

    def set_layer_parameters(self, layer_idx: int, w: np.ndarray, b: np.ndarray):
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"Layer index {layer_idx} out of range [0, {self.num_layers})")

        in_feat = ctypes.c_uint32()
        out_feat = ctypes.c_uint32()
        lib.model_layer_dims(
            self._handle,
            layer_idx,
            ctypes.byref(in_feat),
            ctypes.byref(out_feat),
        )

        w = np.ascontiguousarray(w, dtype=np.float64)
        b = np.ascontiguousarray(b, dtype=np.float64)

        if w.shape != (in_feat.value, out_feat.value):
            raise ValueError(
                f"Weight shape {w.shape} does not match layer dimensions ({in_feat.value}, {out_feat.value})"
            )
        if b.shape != (out_feat.value,):
            raise ValueError(
                f"Bias shape {b.shape} does not match layer output dimension ({out_feat.value},)"
            )

        w_c = w.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        b_c = b.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        lib.model_set_layer_params(self._handle, layer_idx, w_c, b_c)

    def get_layer_gradients(self, layer_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"Layer index {layer_idx} out of range [0, {self.num_layers})")

        in_feat = ctypes.c_uint32()
        out_feat = ctypes.c_uint32()
        lib.model_layer_dims(
            self._handle,
            layer_idx,
            ctypes.byref(in_feat),
            ctypes.byref(out_feat),
        )

        gw = np.empty((in_feat.value, out_feat.value), dtype=np.float64)
        gb = np.empty((out_feat.value,), dtype=np.float64)

        gw_c = gw.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        gb_c = gb.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        lib.model_get_layer_gradients(self._handle, layer_idx, gw_c, gb_c)
        return gw, gb

    def get_dead_neurons(self, X: np.ndarray, layer_idx: int = -1) -> float:
        X = np.ascontiguousarray(X, dtype=np.float64)
        X_c = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        return lib.model_get_dead_neurons(self._handle, X_c, X.shape[0], X.shape[1], layer_idx)

    @property
    def is_training(self) -> bool:
        return lib.model_is_training(self._handle)

    @is_training.setter
    def is_training(self, value: bool):
        lib.model_set_training(self._handle, bool(value))

    def train(self):
        self.is_training = True

    def eval(self):
        self.is_training = False

    def check_gradients(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: float = 1e-5,
        num_checks: int = 5,
    ) -> dict:
        # Temporarily disable layer dropout for deterministic finite differences
        saved_dropouts = [self.get_layer_dropout(i) for i in range(self.num_layers)]
        for i in range(self.num_layers):
            if saved_dropouts[i] > 0.0:
                self.set_layer_dropout(i, 0.0)

        try:
            # Run backward pass with lr=0 to populate analytical gradients without moving weights
            self.train_step(X, y, lr=0.0)

            max_rel_error = 0.0
            checks = []

            for layer_idx in range(self.num_layers):
                w, b = self.get_layer_parameters(layer_idx)
                gw, gb = self.get_layer_gradients(layer_idx)

                # Check random weight entries
                w_coords = [
                    (np.random.randint(0, w.shape[0]), np.random.randint(0, w.shape[1]))
                    for _ in range(num_checks)
                ]
                for r, c in w_coords:
                    w_orig = w[r, c]

                    w[r, c] = w_orig + epsilon
                    self.set_layer_parameters(layer_idx, w, b)
                    l_plus = self.evaluate(X, y)

                    w[r, c] = w_orig - epsilon
                    self.set_layer_parameters(layer_idx, w, b)
                    l_minus = self.evaluate(X, y)

                    w[r, c] = w_orig
                    self.set_layer_parameters(layer_idx, w, b)

                    num_grad = (l_plus - l_minus) / (2.0 * epsilon)
                    ana_grad = gw[r, c]

                    denom = max(1e-8, abs(ana_grad) + abs(num_grad))
                    rel_err = abs(ana_grad - num_grad) / denom
                    max_rel_error = max(max_rel_error, rel_err)

                    checks.append({
                        "param": f"W[{layer_idx}][{r},{c}]",
                        "analytic": ana_grad,
                        "numeric": num_grad,
                        "rel_error": rel_err,
                    })

            return {
                "max_relative_error": max_rel_error,
                "passed": max_rel_error < 1e-5,
                "details": checks,
            }
        finally:
            for i in range(self.num_layers):
                if saved_dropouts[i] > 0.0:
                    self.set_layer_dropout(i, saved_dropouts[i])

    def save_parameters(self, path: str):
        ok = lib.model_save_params(self._handle, path.encode("utf-8"))
        if not ok :
            raise OSError(f"failed to save model to {path}")

    @property
    def _out_dim(self) -> int:
        return lib.model_output_dim(self._handle)

    def __del__(self):
        # Automatically calls Odin free when Python object is garbage collected
        if self._handle:
            lib.model_destroy(self._handle)
            self._handle = None

def load_mlp() -> MLP:
    return MLP(Loss.CCE)
