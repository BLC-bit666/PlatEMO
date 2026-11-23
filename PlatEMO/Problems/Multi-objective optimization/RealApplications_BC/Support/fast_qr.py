"""Exact native rendering and pinned QR decoding with degenerate-trace guards."""
from collections import OrderedDict
import io
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageFilter
import ctypes as ct
from native_kernels import kernels, qr_decoder
from qr_problem import QRImageProblem, PAYLOAD


class FastQRImageProblem(QRImageProblem):
    def __init__(self, cache_size=1024):
        if not isinstance(cache_size, int) or cache_size < 0:
            raise ValueError('cache_size must be a nonnegative integer')
        super().__init__()
        self._kernels = kernels()
        self._decoder = qr_decoder()
        self._guard_start = int(self._decoder.realapps_qr_guard_hits())
        self._payload = PAYLOAD.encode()
        self._decode_error = ct.create_string_buffer(1024)
        self._reference8 = np.asarray(self.reference)
        self._code8 = np.asarray(self.code)
        self._pixels = np.empty_like(self._reference8)
        self.cache_size = cache_size
        self._cache = OrderedDict()
        self._stats = dict(requests=0, cache_hits=0, renders=0, decode_calls=0)

    def statistics(self):
        return dict(self._stats, cache_entries=len(self._cache), cache_limit=self.cache_size,
                    decoder_degenerate_guards=int(self._decoder.realapps_qr_guard_hits())-self._guard_start)

    def _conditions(self, delivered):
        yield delivered
        yield delivered.resize((self.code.width//2, self.code.height//2), Image.Resampling.LANCZOS)
        yield delivered.filter(ImageFilter.GaussianBlur(0.6))

    def evaluate(self, x, save=None, *, short_circuit=False):
        x = np.asarray(x, dtype=float)
        if x.shape != (17,) or not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
            raise ValueError('Decision domain is [0,1]^17')
        start = time.perf_counter(); self._stats['requests'] += 1
        alpha32 = x[:16].reshape(4, 4).astype(np.float32)
        quality = 20 + int(np.rint(75*x[16]))
        # These conversions already occur in the original evaluator. This
        # key adds no rounding and never merges distinguishable renderings.
        key = (alpha32.tobytes(), quality, bool(short_circuit))
        cached = self._cache.get(key)
        if cached is not None:
            self._stats['cache_hits'] += 1
            self._cache.move_to_end(key)
        else:
            alpha_small = Image.fromarray(alpha32)
            alpha = np.asarray(alpha_small.resize(self.code.size, Image.Resampling.BILINEAR))[..., None]
            self._kernels.qr_blend(alpha.size, alpha, self._reference8, self._code8, self._pixels)
            candidate = Image.fromarray(self._pixels)
            stream = io.BytesIO()
            candidate.save(stream, format='JPEG', quality=quality, subsampling=0, optimize=False)
            encoded = stream.getvalue()
            delivered = Image.open(io.BytesIO(encoded)).convert('RGB')
            delivered8 = np.asarray(delivered)
            mse = self._kernels.qr_mse(delivered8.size, delivered8, self._reference8)
            matches = []
            for condition, frame in enumerate(self._conditions(delivered)):
                pixels = delivered8 if condition == 0 else np.asarray(frame)
                # Explicit RGB avoids the Python ndarray API's BGR assumption.
                decoded = self._decoder.realapps_qr_matches(pixels, pixels.shape[1], pixels.shape[0],
                    pixels.strides[0], self._payload, self._decode_error, len(self._decode_error))
                if decoded < 0:
                    raise RuntimeError(self._decode_error.value.decode())
                self._stats['decode_calls'] += 1
                matches.append(bool(decoded))
                if short_circuit and not matches[-1]:
                    break
            self._stats['renders'] += 1
            cached = (encoded, mse, tuple(matches))
            if self.cache_size:
                self._cache[key] = cached
                if len(self._cache)>self.cache_size:
                    self._cache.popitem(last=False)
        encoded, mse, matches = cached
        elapsed = time.perf_counter()-start
        if save:
            Path(save).write_bytes(encoded)
        return dict(x=x.tolist(), objectives=[len(encoded), mse], constraint=int(not all(matches)),
                    decoded_correctly=list(matches), jpeg_quality=quality, seconds=elapsed)
