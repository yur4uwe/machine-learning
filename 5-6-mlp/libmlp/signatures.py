import ctypes
import os

lib_path = os.path.join(os.path.dirname(__file__), "libmlp.so")
lib = ctypes.CDLL(lib_path)

lib.set_seed.argtypes = [ctypes.c_uint64]

lib.model_destroy.argtypes = [ctypes.c_void_p]

lib.model_create.argtypes = [ctypes.c_uint8]
lib.model_create.restype = ctypes.c_void_p

lib.model_add_layer.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint8,
    ctypes.c_uint8,
    ctypes.c_double,
]

lib.model_output_dim.argtypes = [ctypes.c_void_p]
lib.model_output_dim.restype = ctypes.c_uint32

lib.model_num_layers.argtypes = [ctypes.c_void_p]
lib.model_num_layers.restype = ctypes.c_uint32

lib.model_layer_dims.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(ctypes.c_uint32),
]

lib.model_get_layer_params.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]

lib.model_train_step.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
]
lib.model_train_step.restype = ctypes.c_double

lib.model_evaluate.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_double),
]
lib.model_evaluate.restype = ctypes.c_double

lib.model_forward_ffi.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_double),
]

lib.model_save_params.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
]
lib.model_save_params.restype = ctypes.c_bool

lib.model_load_params.argtypes = [
    ctypes.c_char_p,
]
lib.model_load_params.restype = ctypes.c_void_p

lib.model_set_training.argtypes = [
    ctypes.c_void_p,
    ctypes.c_bool,
]

lib.model_is_training.argtypes = [
    ctypes.c_void_p,
]
lib.model_is_training.restype = ctypes.c_bool

lib.model_set_layer_params.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]

lib.model_get_layer_gradients.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]

lib.model_get_dead_neurons.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_int32,
]
lib.model_get_dead_neurons.restype = ctypes.c_double

lib.model_get_hyperparam.argtypes = [ctypes.c_void_p, ctypes.c_uint8]
lib.model_get_hyperparam.restype = ctypes.c_double

lib.model_set_hyperparam.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_double]

lib.model_get_layer_param.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint8]
lib.model_get_layer_param.restype = ctypes.c_double

lib.model_set_layer_param.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.c_uint8,
    ctypes.c_double,
]
