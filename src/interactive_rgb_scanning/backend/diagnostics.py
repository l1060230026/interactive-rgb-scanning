"""Read-only instrumentation of the actual native graph, never a second solver.

Accepted edges come from graph tensor growth, not retrieval print statements.
The deterministic correspondence sample is a diagnostic, not the native robust
objective, a pose ground truth or a geometric-accuracy measurement.
"""

import json
from pathlib import Path
import time

import torch


class NativeGraphDiagnostics:
    def __init__(self, runtime, samples_per_direction=64):
        self.rt = runtime
        self.graph = runtime.native.factor_graph
        self.samples_per_direction = samples_per_direction
        self.events = []
        self.sample_i = []
        self.sample_j = []
        self.sample_pi = []
        self.sample_pj = []
        self.sample_edge = []
        self.seq = None
        self.seconds = 0.0
        self._add = self.graph.add_factors
        self._solve = self.graph.solve_GN_calib
        self.graph.add_factors = self.add_factors
        self.graph.solve_GN_calib = self.solve

    def add_factors(self, ii, jj, min_match_frac, is_reloc=False):
        g = self.graph
        before = len(g.ii)
        # Exactly one original invocation; its return object is passed unchanged.
        result = self._add(ii, jj, min_match_frac, is_reloc=is_reloc)
        tick = time.perf_counter()
        accepted = []
        fractions = []
        for edge in range(before, len(g.ii)):
            i, j = int(g.ii[edge]), int(g.jj[edge])
            accepted.append([i, j])
            frac = []
            for valid, quality, index, a, b in (
                (g.valid_match_j, g.Q_ii2jj, g.idx_ii2jj, i, j),
                (g.valid_match_i, g.Q_jj2ii, g.idx_jj2ii, j, i),
            ):
                good = valid[edge].flatten() & (
                    quality[edge].flatten() > g.cfg["Q_conf"]
                )
                frac.append(float(good.float().mean()))
                pixels = torch.where(good)[0]
                if len(pixels):
                    take = torch.linspace(
                        0,
                        len(pixels) - 1,
                        min(len(pixels), self.samples_per_direction),
                        device=pixels.device,
                    ).long()
                    pj = pixels[take]
                    pi = index[edge].flatten()[pj]
                    self.sample_i.append(torch.full_like(pi, a))
                    self.sample_j.append(torch.full_like(pj, b))
                    self.sample_pi.append(pi)
                    self.sample_pj.append(pj)
                    self.sample_edge.append(torch.full_like(pi, edge))
            fractions.append(dict(edge=[i, j], minimum=min(frac), directions=frac))
        accepted_set = set(map(tuple, accepted))
        requested = [[int(i), int(j)] for i, j in zip(ii, jj)]
        self.events.append(
            dict(
                kind="add_factors",
                seq=self.seq,
                requested=requested,
                accepted=accepted,
                rejected=[e for e in requested if tuple(e) not in accepted_set],
                min_match_frac=float(min_match_frac),
                is_reloc=bool(is_reloc),
                accepted_match_fractions=fractions,
                total_edges=len(g.ii),
            )
        )
        self.seconds += time.perf_counter() - tick
        return result

    def _sample_residuals(self):
        if not self.sample_i:
            return None
        rt = self.rt
        g = self.graph
        i, j, pi, pj = [
            torch.cat(items)
            for items in (self.sample_i, self.sample_j, self.sample_pi, self.sample_pj)
        ]
        poses = (
            rt.native.lietorch.Sim3(
                rt.keyframes.T_WC[: len(rt.keyframes)].reshape(-1, 8)
            )
            .matrix()
            .reshape(-1, 4, 4)
        )
        # Same ray-constrained geometry as native solve_GN_calib. Direction is
        # target pixel pi in camera i <- source point pj in camera j (CUDA code).
        width = rt.w
        K = rt.K
        zi = rt.keyframes.X[i, pi, 2]
        zj = rt.keyframes.X[j, pj, 2]
        rays = torch.stack(
            [
                (pj % width - K[0, 2]) / K[0, 0],
                (pj // width - K[1, 2]) / K[1, 1],
                torch.ones_like(zj),
            ],
            -1,
        )
        local = rays * zj[:, None]
        T = torch.linalg.inv(poses[i]) @ poses[j]
        projected = (T[:, :3, :3] @ local[..., None])[..., 0] + T[:, :3, 3]
        z = projected[:, 2]
        uv = projected[:, :2] / z.clamp_min(1e-8)[:, None] * torch.stack(
            [K[0, 0], K[1, 1]]
        ) + torch.stack([K[0, 2], K[1, 2]])
        target = torch.stack([pi % width, pi // width], -1)
        pixel = (uv - target).norm(dim=-1)
        logdepth = (z.clamp_min(1e-8).log() - zi.clamp_min(1e-8).log()).abs()
        ci = rt.keyframes.C[i, pi, 0] / rt.keyframes.N[i].clamp_min(1)
        cj = rt.keyframes.C[j, pj, 0] / rt.keyframes.N[j].clamp_min(1)
        border = g.cfg["pixel_border"]
        valid = (
            (z > g.cfg["depth_eps"])
            & (zi > g.cfg["depth_eps"])
            & (ci > g.cfg["C_conf"])
            & (cj > g.cfg["C_conf"])
        )
        valid &= (
            torch.isfinite(projected).all(-1)
            & (uv[:, 0] > border)
            & (uv[:, 0] < rt.w - 1 - border)
            & (uv[:, 1] > border)
            & (uv[:, 1] < rt.h - 1 - border)
        )
        return pixel, logdepth, valid

    def solve(self):
        tick = time.perf_counter()
        rt = self.rt
        before = rt.keyframes.T_WC[: len(rt.keyframes)].clone()
        residual_before = self._sample_residuals()
        torch.cuda.synchronize()
        self.seconds += time.perf_counter() - tick
        start = time.perf_counter()
        result = self._solve()  # The original optimizer, unchanged, once.
        torch.cuda.synchronize()
        solve_seconds = time.perf_counter() - start
        tick = time.perf_counter()
        after = rt.keyframes.T_WC[: len(rt.keyframes)].clone()
        residual_after = self._sample_residuals()
        summaries = []
        if residual_before is not None:
            common = residual_before[2] & residual_after[2]
            edge_ids = torch.cat(self.sample_edge)
            for edge in torch.unique(edge_ids).cpu().tolist():
                mask = common & (edge_ids == edge)
                count = int(mask.sum())
                entry = dict(
                    edge=[int(self.graph.ii[edge]), int(self.graph.jj[edge])],
                    common_valid_samples=count,
                )
                for name, res in [
                    ("before", residual_before),
                    ("after", residual_after),
                ]:
                    entry[name] = dict(
                        pixel_median=float(res[0][mask].median()) if count else None,
                        absolute_logdepth_median=(
                            float(res[1][mask].median()) if count else None
                        ),
                    )
                summaries.append(entry)
        self.events.append(
            dict(
                kind="solve_GN_calib",
                seq=self.seq,
                solve_seconds=solve_seconds,
                poses_before=before.cpu().reshape(-1, 8).tolist(),
                poses_after=after.cpu().reshape(-1, 8).tolist(),
                sampled_edge_consistency=summaries,
                residual_scope="Fixed native matched pixels, paired valid intersection, both directions; unweighted sparse pixel/log-depth diagnostics, NOT full native robust objective or accuracy",
            )
        )
        self.seconds += time.perf_counter() - tick
        return result

    def summary(self):
        n = len(self.rt.keyframes)
        edges = list(zip(self.graph.ii.cpu().tolist(), self.graph.jj.cpu().tolist()))
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, j in edges:
            parent[find(i)] = find(j)
        components = {}
        for i in range(n):
            components.setdefault(find(i), []).append(i)
        return dict(
            nodes=n,
            edges=len(edges),
            accepted_nonadjacent_edges=[list(e) for e in edges if abs(e[0] - e[1]) > 1],
            components=list(components.values()),
            solver_calls=sum(e["kind"] == "solve_GN_calib" for e in self.events),
            edge_events=sum(e["kind"] == "add_factors" for e in self.events),
        )

    def export(self, folder):
        folder = Path(folder)
        (folder / "graph-diagnostics.json").write_text(
            json.dumps(dict(summary=self.summary(), events=self.events), indent=2)
            + "\n"
        )
        if self.sample_i:
            self.rt.np.savez_compressed(
                folder / "graph-matched-pixels.npz",
                **{
                    key: torch.cat(value).cpu().numpy()
                    for key, value in [
                        ("i", self.sample_i),
                        ("j", self.sample_j),
                        ("pixel_i", self.sample_pi),
                        ("pixel_j", self.sample_pj),
                        ("edge", self.sample_edge),
                    ]
                },
            )
