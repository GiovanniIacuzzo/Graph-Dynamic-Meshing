#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ENV_FILE="environment.yml"
if [[ ! -f "${ENV_FILE}" ]]; then
    echo "[ERRORE] File ${ENV_FILE} non trovato in ${SCRIPT_DIR}." >&2
    exit 1
fi

ENV_NAME="$(awk -F': *' '/^name:/ {print $2; exit}' "${ENV_FILE}")"
if [[ -z "${ENV_NAME}" ]]; then
    echo "[ERRORE] Impossibile determinare il nome dell'ambiente da ${ENV_FILE}." >&2
    exit 1
fi

echo "[INFO] Ambiente Conda di riferimento: ${ENV_NAME}"

# 1. Verifica disponibilità di conda/mamba
if command -v conda >/dev/null 2>&1; then
    CONDA_CMD="conda"
elif command -v mamba >/dev/null 2>&1; then
    CONDA_CMD="mamba"
else
    echo "[ERRORE] Né 'conda' né 'mamba' sono disponibili nel PATH." >&2
    exit 1
fi

# 2. Creazione o aggiornamento dell'ambiente Conda se non già attivo
if [[ "${CONDA_DEFAULT_ENV:-}" != "${ENV_NAME}" ]]; then
    if "${CONDA_CMD}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
        echo "[INFO] L'ambiente '${ENV_NAME}' esiste già. Aggiornamento dipendenze..."
        "${CONDA_CMD}" env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
    else
        echo "[INFO] Creazione dell'ambiente '${ENV_NAME}' da ${ENV_FILE}..."
        "${CONDA_CMD}" env create -f "${ENV_FILE}"
    fi
    USE_CONDA_RUN=1
else
    echo "[INFO] Ambiente '${ENV_NAME}' già attivo nella sessione corrente."
    USE_CONDA_RUN=0
fi

run_in_env() {
    if [[ "${USE_CONDA_RUN}" == "1" ]]; then
        "${CONDA_CMD}" run --no-capture-output -n "${ENV_NAME}" "$@"
    else
        "$@"
    fi
}

# 3. Verifica degli strumenti di build nell'ambiente
echo "[INFO] Verifica di Python, CMake e pybind11..."
run_in_env python -c "import sys, torch, torch_geometric, pybind11; print(f'Python {sys.version.split()[0]}, PyTorch {torch.__version__}, PyG {torch_geometric.__version__}, pybind11 {pybind11.__version__}')"
run_in_env cmake --version >/dev/null
if command -v ninja >/dev/null 2>&1; then
    echo "[INFO] Ninja disponibile."
else
    echo "[INFO] Ninja non installato: verrà usato il generatore CMake predefinito."
fi

# 4. Verifica di deal.II come dipendenza C++ esterna (opzionale per il solo lato Python, richiesta per _gdm_core)
BUILD_CPP="${GDM_BUILD_CPP:-1}"
if [[ "${BUILD_CPP}" == "1" ]]; then
    if [[ -n "${DEAL_II_DIR:-}" ]]; then
        if [[ ! -d "${DEAL_II_DIR}" ]]; then
            echo "[ERRORE] DEAL_II_DIR è impostata a '${DEAL_II_DIR}', ma la directory non esiste." >&2
            exit 1
        fi
        if [[ ! -f "${DEAL_II_DIR}/lib/cmake/deal.II/deal.IIConfig.cmake" && ! -f "${DEAL_II_DIR}/cmake/config/deal.IIConfig.cmake" && ! -f "${DEAL_II_DIR}/deal.IIConfig.cmake" ]]; then
            echo "[ATTENZIONE] Non è stato trovato deal.IIConfig.cmake nei percorsi standard sotto DEAL_II_DIR='${DEAL_II_DIR}'." >&2
        fi
        echo "[INFO] Utilizzo DEAL_II_DIR=${DEAL_II_DIR}"
    else
        echo "[INFO] Variabile DEAL_II_DIR non impostata: CMake cercherà deal.II nei percorsi di sistema."
        echo "       Se deal.II è installato in un percorso non standard, esporta DEAL_II_DIR=/percorso/a/dealii"
        echo "       oppure imposta GDM_BUILD_CPP=0 per saltare la compilazione dell'estensione C++."
    fi

    BUILD_DIR="${GDM_BUILD_DIR:-build}"
    GENERATOR_ARGS=()
    if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
        CACHED_GENERATOR="$(sed -n 's/^CMAKE_GENERATOR:INTERNAL=//p' "${BUILD_DIR}/CMakeCache.txt" | head -n 1)"
        echo "[INFO] Riutilizzo '${BUILD_DIR}' con il generatore CMake esistente: ${CACHED_GENERATOR:-non specificato}."
        # Do not pass -G to an existing build tree: CMake caches its generator.
    elif command -v ninja >/dev/null 2>&1; then
        GENERATOR_ARGS=(-G Ninja)
    fi

    echo "[INFO] Configurazione e compilazione del core C++ (_gdm_core)..."
    if [[ -n "${DEAL_II_DIR:-}" ]]; then
        run_in_env cmake -S . -B "${BUILD_DIR}" "${GENERATOR_ARGS[@]}" -DCMAKE_BUILD_TYPE=Release "-DDEAL_II_DIR=${DEAL_II_DIR}"
    else
        run_in_env cmake -S . -B "${BUILD_DIR}" "${GENERATOR_ARGS[@]}" -DCMAKE_BUILD_TYPE=Release
    fi
    run_in_env cmake --build "${BUILD_DIR}" --parallel "${GDM_BUILD_JOBS:-2}"
else
    echo "[INFO] GDM_BUILD_CPP=0: compilazione C++ (_gdm_core) saltata."
fi

# 5. Installazione editable del pacchetto Python (senza ricalcolare le dipendenze già gestite da Conda)
echo "[INFO] Installazione del progetto Python in modalità editable..."
run_in_env pip install --no-build-isolation --no-deps -e .

echo "[OK] Ambiente e packaging GDM predisposti con successo (nessuna simulazione o test eseguiti)."