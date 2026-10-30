# Experimental campaigns

The publication workflow is divided into four ordered campaigns. Run commands from the repository root with the `gdm_env` environment active. Scripts are grouped by scientific stage; supporting diagnostics and ablations are kept beside the campaign they support.

| Stage | Purpose | Main entry point | Supporting runs |
|---|---|---|---|
| A | Static Gaussian-field mesh adaptation, target comparison, safety, runtime scaling, OOD and ablations | `python -m experiments.stage_a.campaign` | `python -m experiments.stage_a.evaluate` |
| B | Transient native FEM/ALE and transfer baselines | `python -m experiments.stage_b.campaign` | `python -m experiments.stage_b.diagnostics`, `python -m experiments.stage_b.ablations` |
| C | Repeated end-to-end transient PDE benchmark and transfer-operator comparison | `python -m experiments.stage_c.campaign` | `python -m experiments.stage_c.transfer_benchmark`, `python -m experiments.stage_c.diagnose_transfer` |
| D | Trajectory-aware and self-supervised variants | Scripts under `experiments/stage_d/` | See the Stage-D commands in the root README |

All publication figures must be generated through `python/plotting/` and are exported as vector PDF only. Keep raw per-sample JSON/CSV metrics, configuration snapshots, and logs beside the PDFs so figures can be regenerated without rerunning simulations. Diagnostic runs are not publication evidence until their numerical and architectural tests have passed.
