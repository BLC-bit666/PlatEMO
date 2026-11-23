"""Local exact kernels for the pinned macOS arm64/Python 3.12 runtime."""
import ctypes as ct
from functools import lru_cache
from pathlib import Path
import numpy as np

NATIVE = Path(__file__).resolve().parent/'native'


@lru_cache(maxsize=1)
def kernels():
    lib = ct.CDLL(str(NATIVE/'librealapps_kernels.dylib'))
    array = np.ctypeslib.ndpointer
    f64 = array(dtype=np.float64, flags='C_CONTIGUOUS')
    i32 = array(dtype=np.int32, flags='C_CONTIGUOUS')
    u8 = array(dtype=np.uint8, flags='C_CONTIGUOUS')
    f32 = array(dtype=np.float32, flags='C_CONTIGUOUS')
    lib.route_costs.argtypes = [ct.c_int] + [f64]*6 + [u8, f64]
    lib.route_costs.restype = None
    lib.target_astar.argtypes = [ct.c_int, i32, i32, f64, ct.c_int, ct.c_int, f64, ct.c_double, f64, i32]
    lib.target_astar.restype = ct.c_int
    lib.qr_blend.argtypes = [ct.c_int, f32, u8, u8, u8]
    lib.qr_blend.restype = None
    lib.qr_mse.argtypes = [ct.c_int, u8, u8]
    lib.qr_mse.restype = ct.c_double
    return lib


@lru_cache(maxsize=1)
def qr_decoder():
    # The local reader is the pinned 3.1.1 source plus degenerate-direction
    # guards. Keep the original Python package installed and unchanged.
    import zxingcpp  # Load the pinned Python/C++ runtime before the local bundle.
    lib = ct.CDLL(str(NATIVE/'librealapps_qr.dylib'))
    lib.realapps_qr_matches.argtypes = [np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS'),
        ct.c_int, ct.c_int, ct.c_int, ct.c_char_p, ct.c_void_p, ct.c_int]
    lib.realapps_qr_matches.restype = ct.c_int
    lib.realapps_qr_guard_hits.restype = ct.c_uint64
    return lib
