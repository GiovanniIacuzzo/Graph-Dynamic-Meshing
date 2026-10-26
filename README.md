# Graph Dynamic Meshing (GDM)

Repository per la generazione dei dati, l'addestramento/inferenza della GNN e la ricerca numerica su mesh triangolari mobili. La parte di apprendimento è Python/PyTorch; il core numerico nativo utilizza C++20, deal.II e pybind11.

## Organizzazione

- `dataset/`, `model/`, `utils/`: dataset, architettura GNN e utilità Python.
- `python/geometry/`, `python/numerics/`, `python/oracle/`, `python/evaluation/`: moduli Python geometrici, numerici e di valutazione.
- `python/gdm/`: adapter e API per il core nativo, più runtime ibrido.
- `cpp_core/include/gdm/`, `cpp_core/src/`: interfacce e implementazioni C++.
- `cpp_core/bindings/pybind_module.cpp`: binding del modulo `gdm._gdm_core`.
- `cpp_core/tests/` e `tests/`: test C++ e Python.
- `experiments/`: campagne sperimentali e diagnostiche.

## Prerequisiti

L'ambiente Python è descritto in `environment.yml`. Il progetto richiede Python 3.10 o versione compatibile con le dipendenze dichiarate, NumPy, SciPy, PyTorch, PyTorch Geometric, Matplotlib e PyYAML.

**deal.II 9.8.0 è una dipendenza C++ esterna** e deve essere installata e configurata separatamente. Anche un compilatore C++ compatibile, CMake, Ninja e pybind11 devono essere disponibili. Se deal.II non è in un percorso standard, impostare `DEAL_II_DIR` sulla directory che permette a CMake di individuare `deal.IIConfig.cmake`.

## Preparazione dell'ambiente

Da una shell con Conda o Mamba disponibile:

```bash
bash prepare.sh
```

Lo script crea o aggiorna l'ambiente Conda descritto in `environment.yml`, configura e compila il modulo nativo per impostazione predefinita, quindi installa il progetto Python in modalità editable. Per saltare la build nativa e predisporre solo l'ambiente Python, impostare `GDM_BUILD_CPP=0`; le funzioni che richiedono `_gdm_core` non saranno disponibili finché l'estensione non è compilata.

Per una configurazione manuale del core C++:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Se necessario, aggiungere `-DDEAL_II_DIR=/percorso/di/deal.II` al comando di configurazione. L'estensione per l'installazione editable viene generata nella directory `python/gdm/`.

## Test e validazione

Dopo aver configurato l'ambiente e compilato il core:

```bash
pytest
ctest --test-dir build --output-on-failure
```

Questi comandi sono indicazioni operative; la repository deve essere compilata e i test eseguiti nell'ambiente destinatario prima di considerare verificata la compatibilità con la specifica installazione di deal.II e con la toolchain locale.

## Nota sullo stato della validazione

Le modifiche di manutenzione possono essere ispezionate staticamente, ma la correttezza numerica finale richiede l'esecuzione dei test di regressione, la verifica della D-GCL e la riproduzione dei benchmark previsti. Non sono inclusi in questo documento risultati di test o benchmark non eseguiti.
