"""Shared numerical backend for paired path replays and causal rollouts."""
from __future__ import annotations

import hashlib
from time import perf_counter

import numpy as np
from scipy.integrate import trapezoid

from gdm.native_api import (
    NativeMeshAdapter, apply_native_safety_filter, compute_native_continuous_error_details,
    compute_native_mesh_metrics, create_ale_solver, create_fem_solver, create_pde_model,
)

from .design import ACTIONS, REFERENCE_ACTION, Action, determinants, field_moments
from .design import path_minimum, progress, proposed_end, step_count
from .projection import mass_matrix, project


def mean_in_time(values, times):
    return float(trapezoid(values, times) / (times[-1] - times[0]))


class TransportExperiment:
    def __init__(self, case, cfg, resolution, dt):
        self.case = case
        self.resolution = resolution
        self.dt = dt
        self.duration = float(cfg["time"]["duration"])
        self.horizon = float(cfg["time"]["horizon"])
        self.window_steps = step_count(self.horizon, dt)
        self.windows = step_count(self.duration, self.horizon)
        self.reference = NativeMeshAdapter.from_domain({}, nx=resolution, ny=resolution)
        self.nodes = self.reference.nodes
        self.cells = self.reference.cells
        self.boundary = self.reference.boundary_nodes
        self.reference_j = determinants(self.nodes, self.cells)
        self.motion = cfg["motion"]
        self.floor = float(self.motion["jacobian_ratio"]) * float(self.reference_j.min())
        self.pde = create_pde_model(**case.pde_parameters(self.duration))
        self.solver_options = {
            "stabilization_mode": cfg["solver"]["stabilization"],
            "linear_solver_config": dict(cfg["solver"]["linear"]),
        }
        tick = perf_counter()
        self.ale = create_ale_solver(self.pde, **self.solver_options)
        self.ale_setup_seconds = perf_counter() - tick
        self.initial = self.pde.evaluate_solution(self.nodes, 0.0)

    def state(self, mesh, field):
        nodes = mesh.nodes
        areas = 0.5 * determinants(nodes, self.cells)
        node_mass = np.bincount(self.cells.ravel(), np.repeat(areas / 3.0, 3), minlength=len(nodes))
        return {
            "nodes": nodes,
            "node_mass": node_mass,
            "edges": mesh.directed_edges,
            "boundary": mesh.boundary_mask,
            "field": field,
            "parameters": np.array([self.case.bx, self.case.by, self.case.nu, self.horizon]),
        }

    def endpoint(self, mesh, field, action):
        center, width = field_moments(mesh.nodes, self.cells, field)
        displacement = proposed_end(
            self.nodes, mesh.nodes, self.boundary, center, width,
            (self.case.bx, self.case.by), action, self.horizon,
            float(self.motion["max_speed"]), float(self.motion["strength"]),
        )
        result = apply_native_safety_filter(mesh, displacement, epsilon=self.floor, lock_boundary_nodes=True)
        target = result["safe_nodes"]
        minimum = path_minimum(mesh.nodes, target, self.cells)
        if np.min(minimum) < self.floor - 32 * np.finfo(float).eps:
            raise RuntimeError("The accepted path violated the configured Jacobian floor.")
        return mesh.update_coordinates(target, enforce_fixed_boundary=True), float(result["alpha_global"])

    def measure(self, mesh, field, t):
        diagnostics = compute_native_continuous_error_details(mesh, field, self.pde, t)
        quality = compute_native_mesh_metrics(mesh)
        return {
            "time": t, "l2": float(diagnostics["rel_l2_error"]),
            "h1_seminorm": float(diagnostics["rel_h1_seminorm_error"]),
            "min_jacobian_ratio": float(np.min(determinants(mesh.nodes, self.cells) / self.reference_j)),
            "min_escobar_quality": float(quality["min_escobar_quality"]),
        }

    def relative_peclet(self, mesh, previous_mesh):
        coords = mesh.nodes
        x = coords[self.cells]
        v = (coords - previous_mesh.nodes) / self.dt
        relative = np.asarray([self.case.bx, self.case.by]) - v[self.cells].mean(axis=1)
        sides = np.roll(x, -1, axis=1) - np.roll(x, 1, axis=1)
        grad_phi = np.stack((-sides[:, :, 1], sides[:, :, 0]), axis=-1)
        grad_phi /= determinants(coords, self.cells)[:, None, None]
        directional = np.abs(np.einsum("ei,eji->ej", relative, grad_phi)).sum(axis=1)
        speed_squared = np.einsum("ei,ei->e", relative, relative)
        peclet = np.divide(speed_squared, self.case.nu * directional,
                           out=np.zeros_like(directional), where=directional > 0)
        return float(np.quantile(peclet, .95)), float(peclet.max())

    def advance(self, mesh, field, endpoint, t, timing, update="ale", measure=True, defect=False):
        start = mesh
        rows = []
        projection = {}
        seconds = 0.0
        if update == "remap":
            tick = perf_counter()
            field, projection = project(mesh.nodes, self.cells, field, endpoint.nodes, self.cells)
            # The projection is unconstrained and its mass is measured before
            # the next step imposes the exact Dirichlet trace at the new time.
            mesh = endpoint
            seconds += perf_counter() - tick
        for k in range(1, self.window_steps + 1):
            tick = perf_counter()
            time_next = t + k * self.dt
            if update == "remap" or k == self.window_steps:
                next_mesh = endpoint
            elif start is endpoint:
                next_mesh = mesh
            else:
                fraction = progress(k / self.window_steps, timing)
                next_mesh = start.update_coordinates(
                    start.nodes + fraction * (endpoint.nodes - start.nodes),
                    enforce_fixed_boundary=True,
                )
            new_field, diag = self.ale.step_ale(mesh.core, next_mesh.core, field, self.dt, time_next, False, True)
            seconds += perf_counter() - tick
            if measure:
                row = self.measure(next_mesh, new_field, time_next)
                peclet_q95, peclet_max = self.relative_peclet(next_mesh, mesh)
                row.update(
                    peclet_relative_q95=peclet_q95, peclet_relative_max=peclet_max,
                    gcl_residual=float(diag.algebraic_gcl_residual_linf),
                    linear_residual=float(diag.linear_solver_residual),
                    linear_tolerance=float(diag.linear_solver_tolerance),
                    max_mesh_speed=float(np.linalg.norm(next_mesh.nodes - mesh.nodes, axis=1).max() / self.dt),
                )
                if defect:
                    z_old = self.pde.evaluate_solution(mesh.nodes, time_next - self.dt)
                    z_new = self.pde.evaluate_solution(next_mesh.nodes, time_next)
                    predicted, _ = self.ale.step_ale(mesh.core, next_mesh.core, z_old, self.dt, time_next, False, True)
                    delta = z_new - predicted
                    mass = mass_matrix(next_mesh.nodes, self.cells)
                    row["one_step_state_defect_l2"] = float(np.sqrt(max(0.0, delta @ (mass @ delta))))
                rows.append(row)
            mesh, field = next_mesh, new_field
        return mesh, field, rows, seconds, projection

    def reference_targets(self, mesh, field, t):
        instantaneous, window = [], []
        initial_error = self.measure(mesh, field, t)["l2"]
        endpoints = {}
        for action in ACTIONS:
            if action.anticipation not in endpoints:
                endpoints[action.anticipation] = self.endpoint(mesh, field, action)[0]
            endpoint = endpoints[action.anticipation]
            # Both targets use identical admissible choices and rollout input.
            # The instantaneous objective only sees the reference field at t.
            z = self.pde.evaluate_solution(endpoint.nodes, t)
            instantaneous.append(self.measure(endpoint, z, t)["l2"])
            _, _, rows, _, _ = self.advance(mesh, field, endpoint, t, action.timing)
            window.append(mean_in_time([initial_error, *[row["l2"] for row in rows]],
                                       [t, *[row["time"] for row in rows]]))
        return np.array([instantaneous, window])

    def baseline(self, mode, defect=False, measure=True):
        if mode not in {"fixed_fem", "fixed_ale", "field", "transport"}:
            raise ValueError(f"Unknown baseline: {mode}")
        mesh, field = self.reference, self.initial
        rows = [self.measure(mesh, field, 0.0)] if measure else []
        windows = []
        if mode == "fixed_fem":
            tick = perf_counter()
            fixed = create_fem_solver(mesh, self.pde, **self.solver_options)
            setup = perf_counter() - tick
            for k in range(1, self.windows * self.window_steps + 1):
                t = k * self.dt
                tick = perf_counter()
                field = fixed.step_backward_euler(field, self.dt, t)
                setup += perf_counter() - tick
                if measure:
                    row = self.measure(mesh, field, t)
                    row["peclet_relative_q95"], row["peclet_relative_max"] = self.relative_peclet(mesh, mesh)
                    rows.append(row)
            return rows, windows, setup, field
        if mode == "fixed_ale":
            return self.run_policy(lambda _mesh, _field: None, defect=defect, measure=measure)
        index = 0 if mode == "field" else REFERENCE_ACTION
        return self.run_policy(lambda _mesh, _field: index, defect=defect, measure=measure)

    def run_policy(self, policy, defect=False, measure=True):
        mesh, field = self.reference, self.initial
        rows = [self.measure(mesh, field, 0.0)] if measure else []
        windows, seconds = [], self.ale_setup_seconds
        for w in range(self.windows):
            tick = perf_counter()
            index = policy(mesh, field)
            if index is None:
                endpoint, alpha, timing = mesh, 1.0, 0.0
            else:
                action = ACTIONS[index]
                endpoint, alpha = self.endpoint(mesh, field, action)
                timing = action.timing
            seconds += perf_counter() - tick
            mesh, field, observations, elapsed, _ = self.advance(
                mesh, field, endpoint, w * self.horizon, timing, measure=measure, defect=defect,
            )
            rows.extend(observations)
            windows.append({"window": w, "action": index, "safety_alpha": alpha})
            seconds += elapsed
        return rows, windows, seconds, field

    def paired_paths(self, defect=False, include_remap=True, measure=True):
        # Frozen endpoints from a common causal pilot. The pilot solution
        # is never used as an initial condition for a replay after t=0.
        mesh, field = self.reference, self.initial
        plan = []
        for w in range(self.windows):
            endpoint, alpha = self.endpoint(mesh, field, ACTIONS[REFERENCE_ACTION])
            plan.append((endpoint, alpha))
            mesh, field, _, _, _ = self.advance(mesh, field, endpoint, w * self.horizon, 0, measure=False)
        modes = (("uniform", 0.0, "ale"), ("early", 0.7, "ale"))
        if include_remap:
            modes += (("remap", 0.0, "remap"),)
        results = {}
        for name, timing, update in modes:
            mesh, field = self.reference, self.initial
            rows = [self.measure(mesh, field, 0.0)] if measure else []
            windows, seconds = [], self.ale_setup_seconds
            for w, (endpoint, alpha) in enumerate(plan):
                mesh, field, observations, elapsed, projection = self.advance(
                    mesh, field, endpoint, w * self.horizon, timing, update, measure=measure, defect=defect,
                )
                rows.extend(observations)
                windows.append({
                    "window": w, "safety_alpha": alpha,
                    "endpoint_sha256": hashlib.sha256(endpoint.nodes.astype("<f8").tobytes()).hexdigest(),
                    **projection,
                })
                seconds += elapsed
            results[name] = rows, windows, seconds, field
        return results
