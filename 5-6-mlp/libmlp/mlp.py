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


def set_rng_seed(seed: int):
    lib.set_seed(seed)


@dataclass
class DenseLayer:
    inp_features: int
    out_features: int
    activation: Activation
    init_method: LayerInit = LayerInit.AUTO


class MLP:
    def __init__(self, loss: Loss, *layers: DenseLayer):
        self._handle = lib.model_create(int(loss))
        self._loss = loss
        for layer in layers:
            lib.model_add_layer(
                self._handle,
                layer.inp_features,
                layer.out_features,
                layer.activation,
                layer.init_method,
            )

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

    @property
    def _out_dim(self) -> int:
        return lib.model_output_dim(self._handle)

    def __del__(self):
        # Automatically calls Odin free when Python object is garbage collected
        if self._handle:
            lib.model_destroy(self._handle)
            self._handle = None
