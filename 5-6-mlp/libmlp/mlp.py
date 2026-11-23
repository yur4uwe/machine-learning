import ctypes
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .signatures import lib


class Activation(IntEnum):
    NONE = 0
    RELU = 1
    SOFTMAX = 2


def set_rng_seed(seed: int):
    lib.set_seed(seed)


def cce(preds: np.ndarray, actual: np.ndarray) -> float:
    preds = np.ascontiguousarray(preds, dtype=np.float64)
    actual = np.ascontiguousarray(actual, dtype=np.int32)
    preds_c = preds.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    actual_c = actual.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    return lib.CCE_ffi(preds_c, preds.shape[0], preds.shape[1], actual_c)


@dataclass
class DenseLayer:
    inp_features: int
    out_features: int
    activation: Activation


class MLP:
    def __init__(self, *layers: DenseLayer):
        self._handle = lib.model_create()
        for layer in layers:
            lib.model_add_layer(
                self._handle, layer.inp_features, layer.out_features, layer.activation
            )

    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float) -> float:
        # X and y memory is pinned by NumPy; Odin reads directly via pointer
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.int32)
        X_c = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        y_c = y.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        return lib.model_train_step(self._handle, X_c, X.shape[0], X.shape[1], y_c, lr)

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = np.empty((X.shape[0], self._out_dim), dtype=np.float64)
        X = np.ascontiguousarray(X, dtype=np.float64)
        X_c = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        out_c = out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        lib.model_forward_ffi(self._handle, X_c, X.shape[0], X.shape[1], out_c)
        return out

    @property
    def _out_dim(self) -> int:
        return lib.model_output_dim(self._handle)

    def __del__(self):
        # Automatically calls Odin free when Python object is garbage collected
        if self._handle:
            lib.model_destroy(self._handle)
            self._handle = None
