"""Native calibrated updates with the paper's diagonal regularization."""

import os
from pathlib import Path
import time


def load_solver(upstream, build_directory=None, *, regularized=True):
    from torch.utils.cpp_extension import load

    project = Path(__file__).resolve().parents[3]
    kind = "regularized" if regularized else "native"
    directory = (
        Path(
            build_directory
            or os.environ.get("TORCH_EXTENSIONS_DIR")
            or Path.home() / ".cache/interactive-rgb-scanning/solver"
        )
        / kind
    )
    directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MAX_JOBS", "2")
    return load(
        name="interactive_rgb_solver_" + kind,
        sources=[
            str(project / "csrc/bindings.cpp"),
            str(
                project
                / "csrc"
                / ("gn_kernels.cu" if regularized else "gn_kernels_native.cu")
            ),
        ],
        extra_include_paths=[
            str(Path(upstream) / "mast3r_slam/backend/include"),
            str(Path(upstream) / "thirdparty/eigen"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        build_directory=str(directory),
        verbose=False,
    )


class RegularizedSolver:
    """Install the shared regularized solve and record its numerical status."""

    def __init__(self, runtime, module, primary=None):
        self.runtime, self.module = runtime, module
        self.primary = primary or module
        self.backend = runtime.graph_diagnostics._solve.__func__.__globals__[
            "mast3r_slam_backends"
        ]
        self.original = self.backend.gauss_newton_calib
        self.records = []
        self.backend.gauss_newton_calib = self.solve

    def solve(self, *args):
        before = args[0].clone()
        start = time.perf_counter()
        self.primary.take_solver_probe()
        result = self.primary.gauss_newton_calib(*args)
        status = [int(row[2]) for row in self.primary.take_solver_probe()]
        row = dict(seq=self.runtime.graph_diagnostics.seq, status=status, retried=False)
        if any(status):
            args[0].copy_(before)
            self.module.take_solver_probe()
            result = self.module.gauss_newton_calib(*args)
            row.update(
                retried=True,
                retry_status=[int(r[2]) for r in self.module.take_solver_probe()],
            )
            if any(row["retry_status"]):
                args[0].copy_(before)
                raise RuntimeError(
                    "The calibrated linear solve failed after regularization."
                )
        row["seconds"] = time.perf_counter() - start
        self.records.append(row)
        return result

    def close(self):
        self.backend.gauss_newton_calib = self.original
