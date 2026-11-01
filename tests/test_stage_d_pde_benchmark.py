"""Native deal.II integration tests for the Stage-D PDE-aware trajectory benchmark and audit.

Note: Native tests import ``gdm._gdm_core`` directly without ``importorskip`` so
that any missing deal.II extension causes an explicit test failure rather than
a silent skip.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (ROOT, ROOT / "python"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

import gdm._gdm_core as _gdm_core  # noqa: F401 — fail hard if native deal.II core is not built
from gdm import (
    NativeMeshAdapter,
    apply_native_safety_filter,
    build_gnn_inference_inputs,
    compute_native_continuous_error_details,
    compute_native_discrete_mass,
    compute_native_mesh_metrics,
    compute_native_mesh_metrics_arrays,
    create_ale_solver,
    create_fem_solver,
    create_pde_model,
)
from experiments.stage_d.benchmark_pde import (
    build_inference_engine_for_checkpoint,
    load_stage_d_config,
    resolve_heldout_cases,
    run_pde_trajectory_episode,
)
from experiments.stage_d.audit_benchmark_pde import (
    audit_inference_pipeline_and_raw_proposals,
    compute_paired_bootstrap_ci,
)

CONFIG_PATH = ROOT / "config" / "stage_d.yaml"
STAGE_A_CKPT = ROOT / "results" / "stage_a_oracle_v4" / "checkpoints" / "best_model.pt"
STAGE_D_CKPT = ROOT / "results" / "stage_d_trajectory_aware" / "checkpoints" / "best_model.pt"


def _fast_test_config() -> dict:
    cfg = load_stage_d_config(CONFIG_PATH)
    cfg = copy.deepcopy(cfg)
    cfg["geometry"]["initial_grid"] = {"nx": 10, "ny": 10}
    cfg["stage_d"]["dt"] = 0.005
    cfg["stage_d"]["adapt_interval"] = 5
    cfg["stage_d"]["pde"]["T"] = 0.025
    return cfg


def test_checkpoint_loading_stage_a_and_stage_d() -> None:
    """Verify both Stage A and Stage D checkpoints load cleanly with finite parameters."""
    cfg = load_stage_d_config(CONFIG_PATH)
    assert STAGE_A_CKPT.is_file(), f"Checkpoint Stage A mancante: {STAGE_A_CKPT}"
    assert STAGE_D_CKPT.is_file(), f"Checkpoint Stage D mancante: {STAGE_D_CKPT}"

    engine_a = build_inference_engine_for_checkpoint(STAGE_A_CKPT, cfg, device="cpu")
    engine_d = build_inference_engine_for_checkpoint(STAGE_D_CKPT, cfg, device="cpu")

    for engine in (engine_a, engine_d):
        assert not engine.model.training
        params = list(engine.model.parameters())
        assert len(params) > 0
        for p in params:
            assert torch.all(torch.isfinite(p)), "Trovati pesi non finiti nel checkpoint."


def test_inference_input_consistency_between_stage_a_and_stage_d() -> None:
    """Verify graph construction and input features match between Stage A and Stage D."""
    cfg = _fast_test_config()
    engine_a = build_inference_engine_for_checkpoint(STAGE_A_CKPT, cfg, device="cpu")
    engine_d = build_inference_engine_for_checkpoint(STAGE_D_CKPT, cfg, device="cpu")

    mesh = NativeMeshAdapter.from_domain(cfg["geometry"]["domain"], nx=10, ny=10)
    pde = create_pde_model(bx=0.4, by=0.2, nu=1.0e-3, sigma=0.05, beta=0.45, T=0.025)
    u0 = np.asarray(pde.evaluate_solution(mesh.nodes, 0.0), dtype=np.float64)

    py_mesh, u_in, grad_in = build_gnn_inference_inputs(
        mesh, u0, pde=None, t=0.0, exact_gradient=False
    )
    graph_a = engine_a.prepare_graph(py_mesh, u_in, grad_in)
    graph_d = engine_d.prepare_graph(py_mesh, u_in, grad_in)

    assert graph_a.x.shape == (mesh.num_nodes, 6)
    assert graph_d.x.shape == (mesh.num_nodes, 6)
    assert torch.equal(graph_a.edge_index, graph_d.edge_index)
    assert torch.allclose(graph_a.edge_attr, graph_d.edge_attr, atol=1e-12, rtol=0.0)
    assert torch.allclose(graph_a.x[:, [0, 1, 5]], graph_d.x[:, [0, 1, 5]], atol=1e-12, rtol=0.0)

    pred_a = engine_a.predict_raw(py_mesh, u_in, grad_in)["raw_displacement"]
    pred_d = engine_d.predict_raw(py_mesh, u_in, grad_in)["raw_displacement"]
    assert pred_a.shape == (mesh.num_nodes, 2)
    assert pred_d.shape == (mesh.num_nodes, 2)
    assert np.all(pred_a[mesh.boundary_nodes] == 0.0)
    assert np.all(pred_d[mesh.boundary_nodes] == 0.0)
    assert not np.allclose(pred_a, pred_d), "I modelli Stage A e Stage D devono produrre predizioni distinte."


def test_no_future_or_target_information_leakage() -> None:
    """Verify feature extraction depends exclusively on current mesh and u_h^n."""
    cfg = _fast_test_config()
    engine_d = build_inference_engine_for_checkpoint(STAGE_D_CKPT, cfg, device="cpu")
    mesh = NativeMeshAdapter.from_domain(cfg["geometry"]["domain"], nx=10, ny=10)

    pde_true = create_pde_model(bx=0.4, by=0.2, nu=1.0e-3, sigma=0.05, beta=0.4, T=0.5)
    pde_perturbed = create_pde_model(bx=0.9, by=-0.5, nu=5.0e-2, sigma=0.11, beta=0.9, T=0.5)
    u_curr = np.asarray(pde_true.evaluate_solution(mesh.nodes, 0.05), dtype=np.float64)

    _, u_a, grad_a = build_gnn_inference_inputs(mesh, u_curr, pde=None, t=0.05, exact_gradient=False)
    _, u_b, grad_b = build_gnn_inference_inputs(
        mesh, u_curr, pde=pde_perturbed, t=0.45, exact_gradient=False
    )
    assert np.array_equal(u_a, u_b)
    assert np.allclose(grad_a, grad_b, atol=1e-15, rtol=0.0)

    disp_a = engine_d.predict_raw(mesh, u_a, grad_a)["raw_displacement"]
    disp_b = engine_d.predict_raw(mesh, u_b, grad_b)["raw_displacement"]
    assert np.array_equal(disp_a, disp_b)

    cases = resolve_heldout_cases(cfg, splits=("test_interp", "test_extrap"), smoke_test=True)
    assert len(cases) == 2
    for c in cases:
        assert c["split"] in {"test_interp", "test_extrap"}
        assert "target" not in c and "optimal_nodes" not in c and "displacement" not in c


def test_mesh_update_and_geometric_admissibility_along_stage_d_trajectory() -> None:
    """Verify Stage D GNN updates mesh coordinates and maintains admissibility at every step."""
    cfg = _fast_test_config()
    engine_d = build_inference_engine_for_checkpoint(STAGE_D_CKPT, cfg, device="cpu")
    case = resolve_heldout_cases(cfg, splits=("test_interp",), smoke_test=True)[0]
    case = dict(case, T=0.025, dt=0.005, adapt_interval=5)

    epsilon = float(cfg["geometry"]["safety"]["epsilon"])
    out = run_pde_trajectory_episode(
        case_spec=case,
        strategy="stage_d_gnn",
        config=cfg,
        inference_engine=engine_d,
        compute_diagnostics=True,
    )
    ts_rows = out["timestep_rows"]
    assert len(ts_rows) == 6
    assert out["case_summary"]["final_cumulative_displacement_norm"] > 0.0

    for row in ts_rows:
        assert row["min_jacobian"] >= epsilon - 1.0e-10
        assert row["inadmissible_cells"] == 0
        assert row["inverted_cells"] == 0
        assert row["degenerate_cells"] == 0
        assert 0.0 <= row["safety_alpha"] <= 1.0 + 1.0e-12


def test_native_ale_fem_coupling_and_metric_consistency() -> None:
    """Verify all three strategies execute natively in deal.II and satisfy metric identities."""
    cfg = _fast_test_config()
    engine_a = build_inference_engine_for_checkpoint(STAGE_A_CKPT, cfg, device="cpu")
    engine_d = build_inference_engine_for_checkpoint(STAGE_D_CKPT, cfg, device="cpu")
    case = resolve_heldout_cases(cfg, splits=("test_extrap",), smoke_test=True)[0]
    case = dict(case, T=0.025, dt=0.005, adapt_interval=5)

    outputs = {}
    for strat, eng in (
        ("fixed_fem", None),
        ("stage_a_gnn", engine_a),
        ("stage_d_gnn", engine_d),
    ):
        outputs[strat] = run_pde_trajectory_episode(
            case_spec=case,
            strategy=strat,
            config=cfg,
            inference_engine=eng,
            compute_diagnostics=True,
        )

    step0_fixed = outputs["fixed_fem"]["timestep_rows"][0]
    for strat in ("stage_a_gnn", "stage_d_gnn"):
        step0_dyn = outputs[strat]["timestep_rows"][0]
        assert np.isclose(step0_fixed["relative_l2_error"], step0_dyn["relative_l2_error"], atol=1e-15)
        assert np.isclose(step0_fixed["discrete_mass"], step0_dyn["discrete_mass"], atol=1e-15)
        assert np.isclose(step0_fixed["min_jacobian"], step0_dyn["min_jacobian"], atol=1e-15)

    assert outputs["fixed_fem"]["case_summary"]["fem_solve_time"] > 0.0
    assert outputs["fixed_fem"]["case_summary"]["ale_time"] == 0.0
    assert outputs["fixed_fem"]["case_summary"]["gnn_inference_time"] == 0.0

    for strat in ("stage_a_gnn", "stage_d_gnn"):
        cs = outputs[strat]["case_summary"]
        assert cs["gnn_inference_time"] > 0.0
        assert cs["mesh_update_time"] > 0.0
        assert cs["ale_time"] > 0.0
        assert cs["remapping_time"] == 0.0
        assert np.isfinite(cs["integrated_l2_error"]) and cs["integrated_l2_error"] > 0.0
        assert np.isfinite(cs["final_l2_error"]) and cs["final_l2_error"] > 0.0
        assert np.isfinite(cs["mass_balance_residual"]) and cs["mass_balance_residual"] >= 0.0
        assert cs["gcl_algebraic_residual_linf"] < 1.0e-9


def test_deterministic_reproducibility_of_stage_d_episode() -> None:
    """Verify two consecutive runs of Stage D GNN yield identical numerical trajectories."""
    cfg = _fast_test_config()
    engine_d = build_inference_engine_for_checkpoint(STAGE_D_CKPT, cfg, device="cpu")
    case = resolve_heldout_cases(cfg, splits=("test_interp",), smoke_test=True)[0]
    case = dict(case, T=0.025, dt=0.005, adapt_interval=5)

    run1 = run_pde_trajectory_episode(case, "stage_d_gnn", cfg, engine_d, compute_diagnostics=True)
    run2 = run_pde_trajectory_episode(case, "stage_d_gnn", cfg, engine_d, compute_diagnostics=True)

    assert np.allclose(run1["final_nodes"], run2["final_nodes"], atol=1e-15, rtol=0.0)
    assert np.allclose(run1["final_u"], run2["final_u"], atol=1e-15, rtol=0.0)
    assert np.isclose(
        run1["case_summary"]["integrated_l2_error"],
        run2["case_summary"]["integrated_l2_error"],
        atol=1e-15,
    )


def test_safety_filter_raw_vs_post_filter_metrics_and_exact_alpha_application() -> None:
    """Verify raw inadmissible proposals are contracted to exact admissible post-filter meshes."""
    cfg = _fast_test_config()
    epsilon = float(cfg["geometry"]["safety"]["epsilon"])
    mesh = NativeMeshAdapter.from_domain(cfg["geometry"]["domain"], nx=10, ny=10)

    disp = np.zeros_like(mesh.nodes)
    disp[mesh.interior_mask, 0] = 0.25
    disp[mesh.interior_mask, 1] = -0.22

    raw_geom = compute_native_mesh_metrics_arrays(mesh.nodes + disp, mesh.cells)
    raw_inadm = int(
        np.sum((np.asarray(raw_geom["jacobians"]) < epsilon) | np.asarray(raw_geom["inverted"]))
    )
    assert raw_inadm > 0

    filt = apply_native_safety_filter(
        mesh=mesh,
        displacement=disp,
        epsilon=epsilon,
        safety_margin=1.0e-6,
        tol_deg=1.0e-12,
        lock_boundary_nodes=True,
    )
    alpha = float(filt["alpha_global"])
    safe_nodes = np.asarray(filt["safe_nodes"], dtype=np.float64)
    assert 0.0 < alpha < 1.0
    assert np.allclose(safe_nodes, mesh.nodes + alpha * disp, atol=1e-14, rtol=0.0)

    safe_mesh = mesh.update_coordinates(safe_nodes, enforce_fixed_boundary=True)
    safe_geom = compute_native_mesh_metrics(safe_mesh)
    post_inadm = int(
        np.sum(
            (np.asarray(safe_geom["jacobians"]) < epsilon)
            | np.asarray(safe_geom["inverted"])
            | np.asarray(safe_geom["degenerate"])
        )
    )
    assert post_inadm == 0
    assert int(safe_geom["num_inverted"]) == 0
    assert int(safe_geom["num_degenerate"]) == 0
    assert float(safe_geom["min_jacobian"]) >= epsilon


def test_discrete_mass_quadrature_and_pure_gcl_free_stream_identity() -> None:
    """Verify discrete mass integral consistency across FEM/ALE and pure D-GCL conservation."""
    cfg = _fast_test_config()
    mesh0 = NativeMeshAdapter.from_domain(cfg["geometry"]["domain"], nx=8, ny=8)
    pde = create_pde_model(bx=0.4, by=0.2, nu=1.0e-3, sigma=0.06, beta=0.5, T=0.025)
    fem_solver = create_fem_solver(mesh0, pde)
    # This test isolates the geometric conservation identity from Krylov
    # stopping error.  The direct solve is inexpensive for this 8x8 mesh and
    # leaves the production GMRES defaults unchanged.
    ale_solver = create_ale_solver(
        pde,
        nu=1.0e-3,
        bx=0.4,
        by=0.2,
        linear_solver_config={"use_direct_solver": True},
    )

    u0 = np.asarray(pde.evaluate_solution(mesh0.nodes, 0.0), dtype=np.float64)
    m_api = compute_native_discrete_mass(mesh0, u0)
    m_fem = float(fem_solver.compute_discrete_mass(u0))
    assert abs(m_api - m_fem) < 1.0e-15

    nodes1 = mesh0.nodes.copy()
    x, y = nodes1[:, 0], nodes1[:, 1]
    nodes1[:, 0] += 0.004 * np.sin(np.pi * x) * np.sin(np.pi * y)
    nodes1[:, 1] -= 0.003 * np.sin(np.pi * x) * np.sin(np.pi * y)
    mesh1 = mesh0.update_coordinates(nodes1, enforce_fixed_boundary=True)

    u_const = np.ones(mesh0.num_nodes, dtype=np.float64)
    u_next, diag = ale_solver.step_ale(
        mesh0.core, mesh1.core, u_const, 0.005, 0.005, pure_gcl=True, enforce_dirichlet=False
    )
    free_stream_error = float(np.max(np.abs(np.asarray(u_next) - 1.0)))
    assert free_stream_error < 1.0e-12, (
        "Pure D-GCL free-stream error exceeds tolerance: "
        f"error={free_stream_error:.3e}, "
        f"linear_solver_residual={float(diag.linear_solver_residual):.3e}, "
        f"algebraic_gcl_residual={float(diag.algebraic_gcl_residual_linf):.3e}"
    )
    mass_drift = abs(float(diag.discrete_mass_next) - float(diag.discrete_mass_n))
    assert mass_drift < 1.0e-13, (
        f"Pure D-GCL discrete mass drift={mass_drift:.3e}; "
        f"linear_solver_residual={float(diag.linear_solver_residual):.3e}"
    )
    assert float(diag.algebraic_gcl_residual_linf) < 1.0e-12


def test_timer_mutual_exclusivity_and_ale_fem_solve_time() -> None:
    """Verify exclusive timing components sum to total_wall_time without double counting."""
    cfg = _fast_test_config()
    engine_d = build_inference_engine_for_checkpoint(STAGE_D_CKPT, cfg, device="cpu")
    case = resolve_heldout_cases(cfg, splits=("test_interp",), smoke_test=True)[0]
    case = dict(case, T=0.025, dt=0.005, adapt_interval=5)

    out = run_pde_trajectory_episode(
        case_spec=case,
        strategy="stage_d_gnn",
        config=cfg,
        inference_engine=engine_d,
        compute_diagnostics=False,
    )
    t = out["timings"]
    assert np.isclose(
        t["mesh_update_time"], t["safety_filter_time"] + t["coord_update_time"], atol=1e-15
    )
    pde_time = t.get("ale_fem_solve_time", t["ale_time"] + t["fem_solve_time"])
    assert np.isclose(pde_time, t["ale_time"] + t["fem_solve_time"], atol=1e-15)
    exclusive_sum = (
        t["gnn_inference_time"]
        + t["safety_filter_time"]
        + t["coord_update_time"]
        + pde_time
        + t["remapping_time"]
        + t["other_overhead_time"]
    )
    assert np.isclose(exclusive_sum, t["total_wall_time"], atol=1e-12)


def test_paired_bootstrap_ci_and_audit_pipeline() -> None:
    """Verify paired bootstrap CI helper and source-level inference audit checks."""
    cfg = _fast_test_config()
    audit_res = audit_inference_pipeline_and_raw_proposals(
        cfg=cfg,
        stage_a_checkpoint=STAGE_A_CKPT,
        stage_d_checkpoint=STAGE_D_CKPT,
        device="cpu",
    )
    assert audit_res["call_graph_verified"] is True
    assert audit_res["exact_mms_used_as_gnn_feature"] is False
    assert audit_res["future_states_used_as_gnn_feature"] is False
    assert audit_res["oracle_targets_loaded_in_benchmark"] is False
    assert audit_res["gradient_invariance_to_future_or_perturbed_pde_linf"] == 0.0
    assert audit_res["prediction_invariance_to_future_or_perturbed_pde_linf"] == 0.0

    diffs = [-2.5, -1.8, -3.1, -2.2, -1.9, -2.7]
    stats = compute_paired_bootstrap_ci(diffs, n_bootstrap=2000, confidence=0.95, seed=1701)
    assert stats["n"] == 6
    assert stats["ci95_mean_low"] <= stats["mean"] <= stats["ci95_mean_high"]
    assert stats["ci95_mean_high"] < 0.0