"""
experiments/stage_c_campaign.py
Campagna finale dello Stage C:
1. Esperimento B (End-to-End Closed-Loop Benchmark):
   Confronta Fixed (C0), GDM + P1 (C1), GDM + L2 (C2), GDM + ALE (C3) e Oracle + ALE (C4)
   in esecuzione autonoma closed-loop, misurando direttamente il tempo Wall-Clock end-to-end
   (mediana e IQR su 5 ripetizioni prive di diagnostica) oltre alla scomposizione per componenti.
2. Esperimento A (Controlled Target Replay):
   Confronta P1-Remap, L2-Remap e ALE sulla MEDESIMA sequenza di configurazioni target GNN
   (salvata in formato .npz per completa riproducibilità), riportando separatamente il tempo
   wall-clock di replay e il costo di generazione dei target.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.utils import load_config, ensure_dir
from generate_dataset import create_initial_mesh_2d
from geometry.mesh import TriangularMesh2D
from geometry.geometry import compute_mesh_metrics
from numerics.pde import ManufacturedConvectionDiffusion2D
from numerics.fem import ConvectionDiffusionFEM2D, compute_field_gradients
from numerics.remap import remap_p1_solution
from numerics.remap_l2 import remap_l2_projection
from numerics.dynamic_solver import compute_continuous_errors
from numerics.ale import ALESolver2D
from oracle.field import GaussianBumpField
from oracle.optimizer import OfflineMeshOptimizer
from inference import MeshMoverInferenceEngine


class StageCCampaign:
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        checkpoint_path: str | Path = "results/stage_a/checkpoints/best_model.pt",
        output_dir: str | Path = "results/stage_c",
        n_timing_repeats: int = 5,
    ):
        self.config = load_config(config_path)
        self.output_dir = ensure_dir(output_dir)
        self.figures_dir = ensure_dir(self.output_dir / "figures")
        self.tables_dir = ensure_dir(self.output_dir / "tables")
        self.raw_dir = ensure_dir(self.output_dir / "raw_metrics")
        self.n_timing_repeats = max(1, int(n_timing_repeats))

        self.pde = ManufacturedConvectionDiffusion2D(
            bx=0.4, by=0.2, nu=1e-3, sigma=0.06, beta=0.5, T=0.5
        )
        self.domain = self.config["geometry"]["domain"]

        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint non trovato: {self.checkpoint_path.resolve()}"
            )

        self.T = 0.5
        self.dt = 0.005
        self.adapt_interval = 10
        self.nx = 32
        self.ny = 32
        self.n_steps = int(round(self.T / self.dt))

        self.engine = MeshMoverInferenceEngine(
            checkpoint_path=self.checkpoint_path,
            config=self.config,
            device="cpu",
        )
        self.eps = float(self.config["geometry"]["safety"]["epsilon"])

        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "lines.linewidth": 1.6,
            "grid.alpha": 0.3,
        })

    def _warmup_all_subsystems(self):
        """
        Esegue un warm-up completo di GNN, Safety Filter, solutore FEM, solutore ALE
        e operatori di remapping P1/L2 per eliminare costi di compilazione/allocazione JIT/caches.
        """
        mesh_w = create_initial_mesh_2d(self.domain, nx=self.nx, ny=self.ny)
        u_w = self.pde.evaluate_solution(mesh_w.nodes[:, 0], mesh_w.nodes[:, 1], 0.0)
        grad_w = compute_field_gradients(mesh_w, u_w, self.pde, 0.0, gradient_mode="fem")
        pred_w = self.engine.predict(mesh_w, u_w, grad_w)
        mesh_w2 = pred_w["adapted_mesh"]

        fem_w = ConvectionDiffusionFEM2D(mesh_w, self.pde)
        _ = fem_w.step_backward_euler(u_w, dt=self.dt, t_next=self.dt)

        ale_w = ALESolver2D(self.pde, nu=1e-3, bx=0.4, by=0.2)
        _ = ale_w.step_ale(mesh_w, mesh_w2, u_w, dt=self.dt, t_next=self.dt, pure_gcl=False)

        _ = remap_p1_solution(mesh_w.nodes, mesh_w.cells, u_w, mesh_w2.nodes)
        _ = remap_l2_projection(mesh_w.nodes, mesh_w.cells, u_w, mesh_w2.nodes, mesh_w2.cells)

    # =========================================================================
    # RUNNER UNIFICATO PER SINGOLA CONFIGURAZIONE
    # =========================================================================
    def _execute_single_run(
        self,
        method: str,
        replay_targets: Optional[List[np.ndarray]] = None,
        compute_diagnostics: bool = True,
        record_targets: bool = False,
    ) -> Dict[str, Any]:
        """
        Esegue una singola simulazione su [0, T] con istanti di adattamento sincronizzati
        all'inizio di ciascuna finestra: (step - 1) % adapt_interval == 0.
        Quando compute_diagnostics=False, non esegue alcuna operazione diagnostica né snapshot,
        consentendo una misura wall-clock end-to-end pura.
        """
        init_mesh = create_initial_mesh_2d(self.domain, nx=self.nx, ny=self.ny)

        t_wall_start = time.perf_counter()

        curr_mesh = init_mesh
        u_curr = self.pde.evaluate_solution(curr_mesh.nodes[:, 0], curr_mesh.nodes[:, 1], 0.0)

        fem_time_s = 0.0
        adapt_time_s = 0.0
        safety_time_s = 0.0
        remap_time_s = 0.0

        if method in ["fixed", "gdm_p1", "gdm_l2"]:
            t_f0 = time.perf_counter()
            fem_solver = ConvectionDiffusionFEM2D(curr_mesh, self.pde)
            fem_time_s += time.perf_counter() - t_f0
            ale_solver = None
        else:
            fem_solver = None
            ale_solver = ALESolver2D(self.pde, nu=1e-3, bx=0.4, by=0.2)

        time_arr = [0.0] if compute_diagnostics else []
        l2_series = []
        h1_series = []
        min_j_series = []
        alpha_series = [1.0] if compute_diagnostics else []
        recorded_targets: List[np.ndarray] = []
        adaptation_steps: List[int] = []
        adaptation_times: List[float] = []
        snapshots = [(0.0, curr_mesh.nodes.copy(), u_curr.copy())] if compute_diagnostics else []

        if compute_diagnostics:
            err_l2_0, err_h1_0 = compute_continuous_errors(curr_mesh, u_curr, self.pde, 0.0)
            l2_series.append(err_l2_0)
            h1_series.append(err_h1_0)
            j0 = float(np.min(compute_mesh_metrics(curr_mesh.nodes, curr_mesh.cells)["jacobians"]))
            min_j_series.append(j0)

        base_nodes = curr_mesh.nodes.copy()
        target_nodes = curr_mesh.nodes.copy()
        window_idx = 0
        alpha_current_window = 1.0

        for step in range(1, self.n_steps + 1):
            t_curr = (step - 1) * self.dt
            t_next = step * self.dt

            # 1. Generazione o caricamento del target all'inizio della finestra
            if method != "fixed" and (step - 1) % self.adapt_interval == 0:
                if replay_targets is not None:
                    target_nodes = replay_targets[window_idx]
                    new_mesh = init_mesh.update_coordinates(target_nodes)
                    alpha_current_window = 1.0
                else:
                    if method in ["gdm_p1", "gdm_l2", "gdm_ale"]:
                        t_g0 = time.perf_counter()
                        grad_u = compute_field_gradients(
                            curr_mesh, u_curr, self.pde, t_curr, gradient_mode="fem"
                        )
                        grad_elapsed = time.perf_counter() - t_g0

                        pred = self.engine.predict(curr_mesh, u_curr, grad_u)
                        new_mesh = pred["adapted_mesh"]
                        target_nodes = new_mesh.nodes
                        alpha_current_window = float(pred.get("safety_alpha", 1.0))

                        adapt_time_s += grad_elapsed + float(pred["timing"]["gnn_time"])
                        safety_time_s += float(pred["timing"]["safety_time"])

                    elif method == "oracle_ale":
                        t_or0 = time.perf_counter()
                        cx, cy = self.pde.get_center(t_curr)
                        field_oracle = GaussianBumpField(
                            mu_x=cx,
                            mu_y=cy,
                            sigma=self.pde.sigma,
                            amplitude=np.exp(-self.pde.beta * t_curr),
                        )
                        opt = OfflineMeshOptimizer(
                            reference_field=field_oracle,
                            initial_nodes=curr_mesh.nodes,
                            cells=curr_mesh.cells,
                            boundary_nodes=curr_mesh.boundary_nodes,
                            lambda_D=self.config["oracle"]["objective"]["lambda_D"],
                            lambda_Q=self.config["oracle"]["objective"]["lambda_Q"],
                            epsilon=self.eps,
                            max_iter=3,
                        )
                        opt_res = opt.optimize()
                        target_nodes = opt_res["optimal_nodes"]
                        new_mesh = init_mesh.update_coordinates(target_nodes)
                        alpha_current_window = float(
                            opt_res["final_metrics"].get("safety_alpha_applied", 1.0)
                        )
                        adapt_time_s += time.perf_counter() - t_or0

                if record_targets:
                    recorded_targets.append(target_nodes.copy())
                    adaptation_steps.append(step)
                    adaptation_times.append(t_curr)

                base_nodes = curr_mesh.nodes
                window_idx += 1

                # Se il metodo è a rilocazione discreta (P1 o L2), applica subito il remapping a t_curr
                if method in ["gdm_p1", "gdm_l2"]:
                    t_rm0 = time.perf_counter()

                    if method == "gdm_p1":
                        u_remapped = remap_p1_solution(
                            old_nodes=curr_mesh.nodes,
                            old_cells=curr_mesh.cells,
                            old_u=u_curr,
                            new_nodes=new_mesh.nodes,
                        )
                    else:
                        u_remapped = remap_l2_projection(
                            old_nodes=curr_mesh.nodes,
                            old_cells=curr_mesh.cells,
                            old_u=u_curr,
                            new_nodes=new_mesh.nodes,
                            new_cells=new_mesh.cells,
                        )

                    boundary_nodes = np.asarray(new_mesh.boundary_nodes, dtype=np.int64)
                    u_remapped[boundary_nodes] = self.pde.evaluate_solution(
                        new_mesh.nodes[boundary_nodes, 0],
                        new_mesh.nodes[boundary_nodes, 1],
                        t_curr,
                    )
                    remap_time_s += time.perf_counter() - t_rm0

                    curr_mesh = new_mesh
                    u_curr = u_remapped

                    t_f0 = time.perf_counter()
                    fem_solver = ConvectionDiffusionFEM2D(curr_mesh, self.pde)
                    fem_time_s += time.perf_counter() - t_f0

            # 2. Avanzamento temporale della PDE da t_curr a t_next
            if method in ["fixed", "gdm_p1", "gdm_l2"]:
                t_f0 = time.perf_counter()
                u_curr = fem_solver.step_backward_euler(u_curr, dt=self.dt, t_next=t_next)
                fem_time_s += time.perf_counter() - t_f0
            else:
                k_step = ((step - 1) % self.adapt_interval) + 1
                interp_nodes = base_nodes + (k_step / self.adapt_interval) * (target_nodes - base_nodes)
                next_mesh = init_mesh.update_coordinates(interp_nodes)

                t_f0 = time.perf_counter()
                u_curr, _ = ale_solver.step_ale(
                    curr_mesh,
                    next_mesh,
                    u_curr,
                    dt=self.dt,
                    t_next=t_next,
                    pure_gcl=False,
                )
                fem_time_s += time.perf_counter() - t_f0
                curr_mesh = next_mesh

            # 3. Diagnostica (eseguita SOLO nella corsa diagnostica, mai nelle corse di timing)
            if compute_diagnostics:
                time_arr.append(t_next)
                err_l2, err_h1 = compute_continuous_errors(curr_mesh, u_curr, self.pde, t_next)
                l2_series.append(err_l2)
                h1_series.append(err_h1)

                j_curr = float(np.min(compute_mesh_metrics(curr_mesh.nodes, curr_mesh.cells)["jacobians"]))
                min_j_series.append(j_curr)
                alpha_series.append(alpha_current_window)

                if step in [
                    int(round(self.n_steps / 3)),
                    int(round(2 * self.n_steps / 3)),
                    self.n_steps,
                ]:
                    snapshots.append((t_next, curr_mesh.nodes.copy(), u_curr.copy()))

        total_wall_time_s = time.perf_counter() - t_wall_start
        sum_components_s = fem_time_s + adapt_time_s + safety_time_s + remap_time_s
        other_overhead_s = max(0.0, total_wall_time_s - sum_components_s)

        out: Dict[str, Any] = {
            "fem_time_s": fem_time_s,
            "adapt_time_s": adapt_time_s,
            "safety_time_s": safety_time_s,
            "remap_time_s": remap_time_s,
            "sum_components_s": sum_components_s,
            "other_overhead_s": other_overhead_s,
            "total_wall_time_s": total_wall_time_s,
            "recorded_targets": recorded_targets,
            "adaptation_steps": adaptation_steps,
            "adaptation_times": adaptation_times,
        }

        if compute_diagnostics:
            out.update({
                "time": time_arr,
                "rel_l2_error": l2_series,
                "rel_h1_error": h1_series,
                "final_l2": float(l2_series[-1]),
                "time_int_l2": float(np.trapz(l2_series, time_arr) / self.T),
                "final_h1": float(h1_series[-1]),
                "time_int_h1": float(np.trapz(h1_series, time_arr) / self.T),
                "min_jacobian": float(np.min(min_j_series)),
                "min_j_series": min_j_series,
                "alpha_series": alpha_series,
                "snapshots": snapshots,
                "final_mesh": curr_mesh,
                "final_u": u_curr,
            })

        return out

    def _run_with_clean_timing(
        self,
        method: str,
        replay_targets: Optional[List[np.ndarray]] = None,
        record_targets: bool = False,
    ) -> Dict[str, Any]:
        """
        Esegue:
          1. Una corsa con compute_diagnostics=True per raccogliere errori L2/H1 e metriche geometriche.
          2. n_timing_repeats corse pure (compute_diagnostics=False, record_targets=False)
             misurando direttamente il tempo Wall-Clock end-to-end e i tempi dei singoli sottosistemi,
             calcolando mediana, intervallo interquartile (IQR = Q75 - Q25) e deviazione standard.
        """
        res = self._execute_single_run(
            method,
            replay_targets=replay_targets,
            compute_diagnostics=True,
            record_targets=record_targets,
        )

        repeats = self.n_timing_repeats
        fem_t, adapt_t, safety_t, remap_t, sum_t, ovh_t, wall_t = [], [], [], [], [], [], []

        for _ in range(repeats):
            t_run = self._execute_single_run(
                method,
                replay_targets=replay_targets,
                compute_diagnostics=False,
                record_targets=False,
            )
            fem_t.append(t_run["fem_time_s"])
            adapt_t.append(t_run["adapt_time_s"])
            safety_t.append(t_run["safety_time_s"])
            remap_t.append(t_run["remap_time_s"])
            sum_t.append(t_run["sum_components_s"])
            ovh_t.append(t_run["other_overhead_s"])
            wall_t.append(t_run["total_wall_time_s"])

        q75, q25 = np.percentile(wall_t, [75, 25])

        res["fem_time_s"] = float(np.median(fem_t))
        res["adapt_time_s"] = float(np.median(adapt_t))
        res["safety_time_s"] = float(np.median(safety_t))
        res["remap_time_s"] = float(np.median(remap_t))
        res["sum_components_s"] = float(np.median(sum_t))
        res["other_overhead_s"] = float(np.median(ovh_t))
        res["total_wall_time_s"] = float(np.median(wall_t))
        res["total_time_s"] = res["total_wall_time_s"]  # Alias per compatibilità: ora è il vero wall-clock!
        res["wall_time_iqr_s"] = float(q75 - q25)
        res["wall_time_std_s"] = float(np.std(wall_t))
        res["n_timing_repeats"] = repeats
        return res

    def _save_shared_targets_npz(
        self,
        targets: List[np.ndarray],
        steps: List[int],
        times: List[float],
        cells: np.ndarray,
    ):
        """Salva in un archivio .npz la sequenza di target condivisi dell'Esperimento A per riproducibilità."""
        npz_path = self.raw_dir / "stage_c_shared_targets_exp_a.npz"
        np.savez_compressed(
            npz_path,
            target_nodes=np.asarray(targets, dtype=np.float64),  # shape [n_windows, N, 2]
            adaptation_steps=np.asarray(steps, dtype=np.int64),
            adaptation_times=np.asarray(times, dtype=np.float64),
            cells=np.asarray(cells, dtype=np.int64),
            dt=np.float64(self.dt),
            T=np.float64(self.T),
            adapt_interval=np.int64(self.adapt_interval),
        )
        print(f"[INFO] Sequenza target condivisa (Esperimento A) salvata in {npz_path}")

    def run_benchmark(self):
        print("\n" + "=" * 90)
        print("AVVIO CAMPAGNA STAGE C: ESPERIMENTO B (CLOSED-LOOP) & ESPERIMENTO A (TARGET REPLAY)")
        print("=" * 90)

        print("\n[WARM-UP] Inizializzazione GNN, Safety Filter, FEM, ALE e operatori di Remapping...")
        self._warmup_all_subsystems()

        # ---------------------------------------------------------------------
        # 1. ESPERIMENTO B: Confronto End-to-End Closed-Loop
        # ---------------------------------------------------------------------
        print(f"\n[ESPERIMENTO B] Esecuzione End-to-End Closed-Loop ({self.n_timing_repeats} run wall-clock pure)...")
        closed_loop: Dict[str, Any] = {}

        print("  -> C0: Fixed Eulerian FEM...")
        closed_loop["C0_fixed"] = self._run_with_clean_timing("fixed")

        print("  -> C1: GDM + P1 Pointwise Remap (Closed-Loop)...")
        closed_loop["C1_gdm_p1"] = self._run_with_clean_timing("gdm_p1")

        print("  -> C2: GDM + L2 Projection Remap (Closed-Loop)...")
        closed_loop["C2_gdm_l2"] = self._run_with_clean_timing("gdm_l2")

        print("  -> C3: GDM + ALE D-GCL (Closed-Loop, Metodo proposto)...")
        closed_loop["C3_gdm_ale"] = self._run_with_clean_timing("gdm_ale", record_targets=True)

        print("  -> C4: Oracle + ALE D-GCL (Closed-Loop)...")
        closed_loop["C4_oracle_ale"] = self._run_with_clean_timing("oracle_ale")

        # Salvataggio riproducibile della sequenza di target generata da C3_gdm_ale
        shared_targets = closed_loop["C3_gdm_ale"]["recorded_targets"]
        self._save_shared_targets_npz(
            targets=shared_targets,
            steps=closed_loop["C3_gdm_ale"]["adaptation_steps"],
            times=closed_loop["C3_gdm_ale"]["adaptation_times"],
            cells=closed_loop["C3_gdm_ale"]["final_mesh"].cells,
        )

        # ---------------------------------------------------------------------
        # 2. ESPERIMENTO A: Replay Controllato con Sequenza Target Identica
        # ---------------------------------------------------------------------
        print(f"\n[ESPERIMENTO A] Esecuzione Replay Controllato ({self.n_timing_repeats} run wall-clock pure)...")
        target_gen_cost_s = (
            closed_loop["C3_gdm_ale"]["adapt_time_s"]
            + closed_loop["C3_gdm_ale"]["safety_time_s"]
        )

        replay_results: Dict[str, Any] = {
            "R0_fixed": closed_loop["C0_fixed"],
            "R1_gdm_p1_replay": self._run_with_clean_timing("gdm_p1", replay_targets=shared_targets),
            "R2_gdm_l2_replay": self._run_with_clean_timing("gdm_l2", replay_targets=shared_targets),
            "R3_gdm_ale_replay": self._run_with_clean_timing("gdm_ale", replay_targets=shared_targets),
        }
        for k_rep in ["R1_gdm_p1_replay", "R2_gdm_l2_replay", "R3_gdm_ale_replay"]:
            replay_results[k_rep]["target_gen_reference_s"] = target_gen_cost_s

        self.closed_loop = closed_loop
        self.replay_results = replay_results
        self.snapshots_c3 = closed_loop["C3_gdm_ale"]["snapshots"]
        self.final_mesh_c3 = closed_loop["C3_gdm_ale"]["final_mesh"]

        self._print_summary_tables()
        self._export_results_and_table()
        self._generate_stage_c_figures()

    def _print_summary_tables(self):
        print("\n" + "=" * 112)
        print("RIEPILOGO ESPERIMENTO B — END-TO-END CLOSED-LOOP (Wall-Clock End-to-End Mediana ± IQR su 5 run)")
        print("=" * 112)
        header = (
            f"{'Configurazione':<22} | {'E_L2(T)':<8} | {'Int E_L2':<8} | "
            f"{'min J_e':<9} | {'T_FEM[s]':<8} | {'T_Remap[s]':<10} | {'T_Adapt[s]':<10} | "
            f"{'Wall-Clock [s] (Med ± IQR)':<26}"
        )
        print(header)
        print("-" * len(header))
        for label, key in [
            ("Fixed FEM (C0)", "C0_fixed"),
            ("GDM + P1 Remap (C1)", "C1_gdm_p1"),
            ("GDM + L2 Remap (C2)", "C2_gdm_l2"),
            ("GDM + ALE (C3, Ours)", "C3_gdm_ale"),
            ("Oracle + ALE (C4)", "C4_oracle_ale"),
        ]:
            r = self.closed_loop[key]
            adapt_tot = r["adapt_time_s"] + r["safety_time_s"]
            wc_str = f"{r['total_wall_time_s']:.3f} ± {r['wall_time_iqr_s']:.3f}"
            print(
                f"{label:<22} | {r['final_l2']*100:<7.2f}% | {r['time_int_l2']*100:<7.2f}% | "
                f"{r['min_jacobian']:<9.2e} | {r['fem_time_s']:<8.3f} | {r['remap_time_s']:<10.3f} | "
                f"{adapt_tot:<10.3f} | {wc_str:<26}"
            )

        s_wall_l2 = (
            self.closed_loop["C2_gdm_l2"]["total_wall_time_s"]
            / self.closed_loop["C3_gdm_ale"]["total_wall_time_s"]
        )
        s_comp_l2 = (
            self.closed_loop["C2_gdm_l2"]["sum_components_s"]
            / self.closed_loop["C3_gdm_ale"]["sum_components_s"]
        )
        print("-" * len(header))
        print(f"Speedup End-to-End Wall-Clock (GDM+ALE vs GDM+L2) : {s_wall_l2:.2f}x")
        print(f"Speedup Somma Componenti      (GDM+ALE vs GDM+L2) : {s_comp_l2:.2f}x")

        print("\n" + "=" * 112)
        print("RIEPILOGO ESPERIMENTO A — CONTROLLED TARGET REPLAY (Sequenza target precalcolata condivisa)")
        print("=" * 112)
        header_a = (
            f"{'Configurazione':<22} | {'E_L2(T)':<8} | {'Int E_L2':<8} | "
            f"{'min J_e':<9} | {'T_FEM[s]':<8} | {'T_Remap[s]':<10} | "
            f"{'Replay Wall-Clock [s]':<22} | {'Gen. Target Ref [s]':<19}"
        )
        print(header_a)
        print("-" * len(header_a))
        for label, key in [
            ("Fixed FEM (R0)", "R0_fixed"),
            ("GDM + P1 (Replay R1)", "R1_gdm_p1_replay"),
            ("GDM + L2 (Replay R2)", "R2_gdm_l2_replay"),
            ("GDM + ALE (Replay R3)", "R3_gdm_ale_replay"),
        ]:
            r = self.replay_results[key]
            wc_str = f"{r['total_wall_time_s']:.3f} ± {r['wall_time_iqr_s']:.3f}"
            gen_ref = r.get("target_gen_reference_s", 0.0)
            print(
                f"{label:<22} | {r['final_l2']*100:<7.2f}% | {r['time_int_l2']*100:<7.2f}% | "
                f"{r['min_jacobian']:<9.2e} | {r['fem_time_s']:<8.3f} | {r['remap_time_s']:<10.3f} | "
                f"{wc_str:<22} | {gen_ref:<19.3f}"
            )
        s_rep_wall = (
            self.replay_results["R2_gdm_l2_replay"]["total_wall_time_s"]
            / self.replay_results["R3_gdm_ale_replay"]["total_wall_time_s"]
        )
        print("-" * len(header_a))
        print(f"Speedup Replay Wall-Clock Puro (ALE vs L2, a target già noti): {s_rep_wall:.2f}x")

    def _export_results_and_table(self):
        exclude_keys = {"time", "snapshots", "final_mesh", "final_u", "recorded_targets"}
        payload = {
            "experiment_b_closed_loop": {
                k: {m: v for m, v in val.items() if m not in exclude_keys}
                for k, val in self.closed_loop.items()
            },
            "experiment_a_target_replay": {
                k: {m: v for m, v in val.items() if m not in exclude_keys}
                for k, val in self.replay_results.items()
            },
        }
        with open(self.raw_dir / "stage_c_benchmark_results.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        table_path = self.tables_dir / "table_stage_c_summary.tex"
        rows = [
            ("Fixed FEM (C0)", "C0_fixed"),
            ("GDM + $P_1$ (C1)", "C1_gdm_p1"),
            ("GDM + $L^2$ (C2)", "C2_gdm_l2"),
            ("GDM + ALE (C3)", "C3_gdm_ale"),
            ("Oracle + ALE (C4)", "C4_oracle_ale"),
        ]

        t_l2_wall = self.closed_loop["C2_gdm_l2"]["total_wall_time_s"]

        latex_rows = []
        for name, k in rows:
            r = self.closed_loop[k]
            adapt_s = r["adapt_time_s"] + r["safety_time_s"]
            sp_l2 = t_l2_wall / r["total_wall_time_s"]
            latex_rows.append(
                f"{name} & {r['final_l2']*100:.2f}\\% & {r['time_int_l2']*100:.2f}\\% & "
                f"{r['min_jacobian']:.2e} & {r['fem_time_s']:.3f} & {r['remap_time_s']:.3f} & "
                f"{adapt_s:.3f} & {r['total_wall_time_s']:.3f} ($\\pm${r['wall_time_iqr_s']:.3f}) & "
                f"{sp_l2:.2f}$\\times$ \\\\"
            )

        latex_table = (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            "\\footnotesize\n"
            "\\caption{Benchmark dinamico end-to-end in closed-loop (Esperimento B, $T=0.5, \\Delta t=0.005$, mesh $32\\times 32$, mediana $\\pm$ IQR su $5$ esecuzioni).}\n"
            "\\label{tab:stage_c_summary}\n"
            "\\begin{tabular}{lcccccccc}\n"
            "\\toprule\n"
            "Configurazione & $E_{L^2}(T)$ & $\\overline{E}_{L^2}$ & $\\min_{t,e} J_e$ & "
            "$T_{\\mathrm{FEM/ALE}}$ [s] & $T_{\\mathrm{Remap}}$ [s] & $T_{\\mathrm{Adapt}}$ [s] & "
            "$T_{\\mathrm{wall}}$ [s] & Speedup vs $L^2$ \\\\\n"
            "\\midrule\n"
            + "\n".join(latex_rows)
            + "\n\\bottomrule\n"
            "\\end{tabular}\n"
            "\\end{table}\n"
        )
        with open(table_path, "w", encoding="utf-8") as f:
            f.write(latex_table)
        print(f"\n[INFO] Tabella salvata in {table_path}")

    def _generate_stage_c_figures(self):
        print("\n---> Generazione Figure Scientifiche dello Stage C...")
        t_arr = self.closed_loop["C0_fixed"]["time"]

        # FIGURA C3: Evoluzione Temporale dell'Errore PDE (Benchmark Finale)
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.plot(
            t_arr,
            np.array(self.closed_loop["C0_fixed"]["rel_l2_error"]) * 100,
            label="Fixed FEM (C0)",
            color="#555555",
            linestyle="--",
        )
        ax.plot(
            t_arr,
            np.array(self.closed_loop["C1_gdm_p1"]["rel_l2_error"]) * 100,
            label=r"GDM + $P_1$ Remap (C1)",
            color="#d62728",
            linestyle="-.",
        )
        ax.plot(
            t_arr,
            np.array(self.closed_loop["C2_gdm_l2"]["rel_l2_error"]) * 100,
            label=r"GDM + $L^2$ Projection (C2)",
            color="#ff7f0e",
        )
        ax.plot(
            t_arr,
            np.array(self.closed_loop["C3_gdm_ale"]["rel_l2_error"]) * 100,
            label="GDM + ALE (C3, Proposed)",
            color="#1f77b4",
            linewidth=2.0,
        )
        ax.plot(
            t_arr,
            np.array(self.closed_loop["C4_oracle_ale"]["rel_l2_error"]) * 100,
            label="Oracle + ALE (C4)",
            color="#2ca02c",
            linestyle=":",
        )

        ax.set_xlabel("Time $t$ [s]")
        ax.set_ylabel(r"Continuous Relative $L^2$ Error [%]")
        ax.grid(True)
        ax.legend(frameon=True, fontsize=9)
        fig.tight_layout()
        fig.savefig(self.figures_dir / "fig_c3_dynamic_error.png", dpi=300)
        plt.close(fig)

        # FIGURA C4: Evoluzione del Minimo Determinante Jacobiano min_e J_e(t)
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.plot(
            t_arr,
            self.closed_loop["C2_gdm_l2"]["min_j_series"],
            label=r"GDM + $L^2$ $\min_e J_e(t)$",
            color="#ff7f0e",
            linestyle="--",
        )
        ax.plot(
            t_arr,
            self.closed_loop["C3_gdm_ale"]["min_j_series"],
            label=r"GDM + ALE $\min_e J_e(t)$",
            color="#1f77b4",
        )
        ax.plot(
            t_arr,
            self.closed_loop["C4_oracle_ale"]["min_j_series"],
            label=r"Oracle + ALE $\min_e J_e(t)$",
            color="#2ca02c",
            linestyle=":",
        )
        ax.axhline(
            self.eps,
            color="#d62728",
            linestyle=":",
            linewidth=1.5,
            label=rf"Safety Threshold $\epsilon = {self.eps:.1e}$",
        )
        ax.set_yscale("log")
        ax.set_xlabel("Time $t$ [s]")
        ax.set_ylabel(r"Minimum Jacobian $\min_e J_e$")
        ax.grid(True, which="both")
        ax.legend(frameon=True, fontsize=9)
        fig.tight_layout()
        fig.savefig(self.figures_dir / "fig_c4_mesh_validity.png", dpi=300)
        plt.close(fig)

        # FIGURA C5: Decomposizione Algoritmica e Wall-Clock del Runtime
        fig, ax = plt.subplots(figsize=(6.8, 4.0))
        cfgs = ["Fixed FEM", r"GDM + $P_1$", r"GDM + $L^2$", "GDM + ALE (Ours)"]
        keys = ["C0_fixed", "C1_gdm_p1", "C2_gdm_l2", "C3_gdm_ale"]

        fem_times = np.array([self.closed_loop[k]["fem_time_s"] for k in keys])
        remap_times = np.array([self.closed_loop[k]["remap_time_s"] for k in keys])
        adapt_times = np.array([
            self.closed_loop[k]["adapt_time_s"] + self.closed_loop[k]["safety_time_s"]
            for k in keys
        ])
        ovh_times = np.array([self.closed_loop[k]["other_overhead_s"] for k in keys])

        x = np.arange(len(cfgs))
        width = 0.48
        ax.bar(x, fem_times, width, label="FEM/ALE Assembly & Solve", color="#1f77b4")
        ax.bar(x, remap_times, width, bottom=fem_times, label="State Remapping", color="#d62728")
        ax.bar(
            x,
            adapt_times,
            width,
            bottom=fem_times + remap_times,
            label="GNN + Safety Filter",
            color="#2ca02c",
        )
        ax.bar(
            x,
            ovh_times,
            width,
            bottom=fem_times + remap_times + adapt_times,
            label="Mesh Update & Loop Overhead",
            color="#9467bd",
        )

        ax.set_ylabel("Wall-Clock Execution Time [s]")
        ax.set_xticks(x)
        ax.set_xticklabels(cfgs)
        ax.grid(True, axis="y")
        ax.legend(frameon=True, fontsize=9)
        fig.tight_layout()
        fig.savefig(self.figures_dir / "fig_c5_runtime_decomposition.png", dpi=300)
        plt.close(fig)

        # FIGURA C6: Snapshot Qualitativi e Tracciamento della Soluzione
        fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.0))
        for col, (snap_t, snap_nodes, snap_u) in enumerate(self.snapshots_c3):
            tri = mtri.Triangulation(snap_nodes[:, 0], snap_nodes[:, 1], self.final_mesh_c3.cells)
            axes[0, col].triplot(tri, "b-", lw=0.4)
            axes[0, col].set_title(f"Mesh: $t = {snap_t:.2f}$ s")

            axes[1, col].tricontourf(tri, snap_u, levels=25, cmap="viridis")
            axes[1, col].set_title(f"ALE $u_h(t={snap_t:.2f})$")

            for row in range(2):
                axes[row, col].set_aspect("equal")
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])

        fig.tight_layout()
        fig.savefig(self.figures_dir / "fig_c6_qualitative_tracking.png", dpi=300)
        plt.close(fig)

        print(f"[INFO] Figure generate in {self.figures_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark dinamico Stage C per Graph-Dynamic Meshing"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--checkpoint", default="results/stage_a/checkpoints/best_model.pt")
    parser.add_argument("--output_dir", default="results/stage_c")
    parser.add_argument("--repeats", type=int, default=5, help="Ripetizioni per mediana/IQR wall-clock")
    args = parser.parse_args()

    campaign = StageCCampaign(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        n_timing_repeats=args.repeats,
    )
    campaign.run_benchmark()


if __name__ == "__main__":
    main()
