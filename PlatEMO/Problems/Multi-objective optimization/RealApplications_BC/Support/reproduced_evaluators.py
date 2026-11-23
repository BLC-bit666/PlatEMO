"""Batch transport; FE counts requests, cache/solver counts are separate."""
import importlib
import sys
from pathlib import Path

import numpy as np


class Evaluator:
    def __init__(self, artifact_root, kind):
        code = str(Path(artifact_root) / 'code')
        if code not in sys.path:
            sys.path.insert(0, code)
        names = {'qr': ('fast_qr', 'FastQRImageProblem'),
                 'route': ('fast_route', 'FastBlackForestRouteProblem')}
        module, constructor = names[str(kind)]
        self.kind = str(kind)
        self.problem = getattr(importlib.import_module(module), constructor)()
        self.calls = 0

    def statistics(self):
        return dict(logical_calls=self.calls, problem=self.kind, **self.problem.statistics())

    def evaluate(self, decisions):
        x = np.asarray(decisions, dtype=np.float64).reshape(-1, self.problem.dimension)
        result = np.empty((len(x), 3), dtype=np.float64)
        for i, row in enumerate(x):
            evaluated = (self.problem.evaluate(row, short_circuit=True) if self.kind == 'qr'
                         else self.problem.evaluate(row))
            result[i, :2] = evaluated['objectives']
            result[i, 2] = evaluated['constraint']
            self.calls += 1
        return result
