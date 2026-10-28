"""Backward-compatible entry point for the canonical native Stage-C benchmark.

The old file duplicated FEM/ALE/remapping kernels in Python. The maintained
operator-level remapping benchmark now lives in ``stage_c_transfer_operator_benchmark.py``;
this entry point forwards to the canonical end-to-end native Stage-C campaign.
"""
from __future__ import annotations

from experiments.stage_c_campaign import main


if __name__ == "__main__":
    main()
