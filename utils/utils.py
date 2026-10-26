"""
utils/utils.py
Funzioni di utilità trasversali e agnostiche per Graph-Dynamic Meshing (GDM).

Fornisce:
    - set_seed: determinismo e gestione dei generatori pseudocasuali (Python, NumPy, PyTorch)
    - ensure_dir: creazione sicura di directory nel filesystem
    - load_config / save_config: I/O robusto per configurazioni YAML
    - save_pickle / load_pickle: serializzazione e deserializzazione binaria di oggetti Python
    - Timer: context manager per la profilazione cronometrica a precisione nanosecondo
    - setup_logger: inizializzazione del sistema di logging su console e file
"""

from typing import Any, Dict, Optional, Union
from pathlib import Path
import os
import sys
import time
import random
import pickle
import logging
import yaml
import numpy as np


# =============================================================================
# 1. RIPRODUCIBILITÀ E SEED MANAGEMENT
# =============================================================================

def set_seed(seed: int = 42, deterministic_torch: bool = True) -> None:
    """
    Fissa il seme dei generatori di numeri pseudocasuali per garantire
    la riproducibilità rigorosa degli esperimenti scientifici.

    Input:
        seed: valore intero del seme
        deterministic_torch: se True, impone flag di esecuzione deterministica in PyTorch
    """
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # Inizializzazione condizionale se PyTorch è presente nell'ambiente
    if "torch" in sys.modules:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            if deterministic_torch:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        elif torch.backends.mps.is_available():
            # MPS usa manual_seed di torch
            pass


# =============================================================================
# 2. GESTIONE FILESYSTEM E DIRECTORY
# =============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Verifica l'esistenza di un percorso cartella e, se assente, crea la struttura
    completa dei genitori (mkdir -p).

    Input:
        path: stringa o oggetto Path della directory da creare

    Output:
        oggetto Path risolto
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# =============================================================================
# 3. CONFIGURAZIONE E SERIALIZZAZIONE (YAML / PICKLE)
# =============================================================================

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Carica e analizza un file di configurazione in formato YAML.

    Input:
        config_path: percorso al file .yaml

    Output:
        dizionario contenente la gerarchia dei parametri
    """
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"File di configurazione non trovato: {p.resolve()}")

    with open(p, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Il file YAML deve contenere una mappatura chiave-valore, ottenuto: {type(config)}")

    return config


def save_config(config: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Salva un dizionario di configurazione in un file YAML formattato.

    Input:
        config: dizionario dei parametri
        output_path: percorso di destinazione del file .yaml
    """
    p = Path(output_path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


def save_pickle(obj: Any, file_path: Union[str, Path]) -> None:
    """
    Serializza un oggetto generico su disco tramite protocollo binario pickle.
    """
    p = Path(file_path)
    ensure_dir(p.parent)
    with open(p, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(file_path: Union[str, Path]) -> Any:
    """
    Carica e deserializza un oggetto binario salvato in precedenza via pickle.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File pickle non trovato: {p.resolve()}")
    with open(p, "rb") as f:
        return pickle.load(f)


# =============================================================================
# 4. CRONOMETRAGGIO E PROFILAZIONE (TIMING HELPERS)
# =============================================================================

class Timer:
    """
    Context manager e oggetto cronometro per misurare intervalli di esecuzione.
    
    Utilizzo come context manager:
        with Timer() as t:
            # esecuzione codice
        print(f"Tempo trascorso: {t.elapsed_seconds:.4f} s")
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed_seconds: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_time = time.perf_counter()
        self.elapsed_seconds = self.end_time - self.start_time

    @property
    def elapsed_ms(self) -> float:
        """Restituisce il tempo trascorso in millisecondi."""
        return self.elapsed_seconds * 1000.0


# =============================================================================
# 5. SISTEMA DI LOGGING CENTRALIZZATO
# =============================================================================

def setup_logger(
    name: str = "GDM",
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Inizializza un logger strutturato per la tracciabilità su console e file.

    Input:
        name: etichetta identificativa del logger
        log_file: percorso opzionale dove archiviare il file di testo di log
        level: soglia minima di gravità (default logging.INFO)

    Output:
        istanza configurata di logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Evita la registrazione ridondante di handler se già presente
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler su standard output (console)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # Handler su file (se specificato)
    if log_file is not None:
        p = Path(log_file)
        ensure_dir(p.parent)
        file_handler = logging.FileHandler(p, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    return logger

# =============================================================================
# 6. DEVICE AND CHECKPOINT HELPERS
# =============================================================================

def get_device(device: Union[str, "Any"] = "auto"):
    """Resolve a PyTorch device string with explicit availability checks.

    ``auto`` prefers CUDA, then Apple MPS, then CPU. An explicitly requested
    accelerator that is unavailable raises an actionable error rather than
    silently moving the experiment to a different device.
    """
    import torch

    if isinstance(device, torch.device):
        name = device.type
        requested = device
    else:
        name = str(device).strip().lower()
        requested = None

    if name in {"", "auto", "default"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested is not None:
        if name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("È stato richiesto CUDA, ma CUDA non è disponibile in questa installazione di PyTorch.")
        if name == "mps":
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is None or not mps_backend.is_available():
                raise RuntimeError("È stato richiesto MPS, ma il backend Apple MPS non è disponibile.")
        return requested

    if name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"È stato richiesto il dispositivo {name!r}, ma CUDA non è disponibile.")
        try:
            return torch.device(name)
        except Exception as exc:
            raise ValueError(f"Dispositivo PyTorch non valido: {device!r}.") from exc
    if name == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError("È stato richiesto MPS, ma il backend Apple MPS non è disponibile.")
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    try:
        return torch.device(name)
    except Exception as exc:
        raise ValueError(f"Dispositivo PyTorch non riconosciuto: {device!r}.") from exc


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: Any,
    optimizer: Optional[Any] = None,
    device: Optional[Any] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load a training checkpoint into a model and optionally an optimizer.

    Supports the repository's structured checkpoints (``model_state_dict``)
    and raw state dictionaries. Returns the complete checkpoint payload.
    """
    import torch

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint non trovato: {path.resolve()}")
    map_location = device if device is not None else "cpu"
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # Compatibilità con versioni di PyTorch senza weights_only.
        payload = torch.load(path, map_location=map_location)

    if not isinstance(payload, dict):
        raise ValueError("Il checkpoint deve contenere un dizionario o uno state_dict.")

    state_dict = payload.get("model_state_dict")
    if state_dict is None:
        state_dict = payload.get("state_dict")
    if state_dict is None and payload and all(isinstance(k, str) for k in payload):
        # Un raw state_dict è composto da nomi di parametri e tensori, non da
        # metadati (epoch/config/optimizer_state_dict).
        if not any(k in payload for k in ("epoch", "config", "optimizer_state_dict", "best_val_loss")):
            state_dict = payload
    if not isinstance(state_dict, dict):
        raise KeyError("Checkpoint privo di 'model_state_dict', 'state_dict' o raw state_dict riconoscibile.")

    model.load_state_dict(state_dict, strict=bool(strict))
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload
