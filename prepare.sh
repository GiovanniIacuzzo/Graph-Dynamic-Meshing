#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Use the active Python environment and a matching deal.II installation.
# Pass extra CMake options directly to this script, including -DDEAL_II_DIR.
python -m pip install -e '.[dev]'
cmake -S . -B "${GDM_BUILD_DIR:-build}" -DCMAKE_BUILD_TYPE=Release \
    "-DPython3_EXECUTABLE=$(python -c 'import sys; print(sys.executable)')" \
    "-Dpybind11_DIR=$(python -m pybind11 --cmakedir)" "$@"
cmake --build "${GDM_BUILD_DIR:-build}" --parallel "${GDM_BUILD_JOBS:-2}"
