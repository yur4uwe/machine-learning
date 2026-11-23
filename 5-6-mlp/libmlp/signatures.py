import ctypes
import os

lib_path = os.path.join(os.path.dirname(__file__), "libmlp.so")
lib = ctypes.CDLL(lib_path)

lib.set_seed.argtypes = [ctypes.c_uint64]

lib.model_destroy.argtypes = [ctypes.c_void_p]
lib.model_create.restype = ctypes.c_void_p

lib.model_add_layer.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint8,
    ctypes.c_uint8,
]

lib.model_output_dim.argtypes = [ctypes.c_void_p]
lib.model_output_dim.restype = ctypes.c_uint32

lib.model_train_step.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.c_double,
]
lib.model_train_step.restype = ctypes.c_double

lib.model_forward_ffi.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_double),
]

lib.CCE_ffi.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_int32),
]
lib.CCE_ffi.restype = ctypes.c_double
