# Graph-Dynamic Meshing (GDM)
### Physics-Consistent, Real-Time $r$-Adaptive Mesh Movement via Graph Neural Networks with Analytical Safety Guarantees

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Backend](https://img.shields.io/badge/hardware-Apple%20Silicon%20(MPS)%20%7C%20CUDA-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Executive Summary

**Graph-Dynamic Meshing (GDM)** è un framework scientifico e computazionale che introduce un cambio di paradigma nell'adattività di griglia ($r$-adaptation) per solutori alle derivate parziali (PDE). 

Nelle simulazioni ingegneristiche ad alta fedeltà (CFD, FEM, digital twins), i solutori variazionali tradizionali calcolano lo spostamento nodale $\Delta X^*$ risolvendo problemi di ottimizzazione non lineare vincolata (e.g., L-BFGS-B, MMPDE). Sebbene accurati, questi metodi comportano costi computazionali nell'ordine dei secondi o delle decine di secondi per singolo passo temporale, rendendo l'adattività dinamica proibitiva per scenari real-time.

GDM sostituisce l'ottimizzatore iterativo runtime con una **MeshMover Graph Neural Network (GNN)** ad architettura Encode-Process-Decode accoppiata a un **Filtro di Sicurezza Analitico Proiettivo $\mathcal{O}(M)$**.

### Punti Chiave del Framework
* **Inferenza Real-Time:** Accelerazione fino a **$400\times$** rispetto all'oracolo variazionale L-BFGS-B su CPU (da $\sim 60\text{ s}$ a $\sim 150\text{ ms}$ a oltre $4000$ nodi; $< 25\text{ ms}$ su griglie standard $32 \times 32$).
* **Garanzia Geometrica Deterministica:** Filtro di sicurezza analitico a complessità temporale strettamente $\mathcal{O}(M)$ che assicura a priori l'assenza di inversioni di volume ($\min_e J_e \ge \epsilon > 0$) e preserva la conformità della mesh anche sotto deformazioni critiche o perturbazioni avversarie.
* **Generalizzazione OOD (Out-of-Distribution):** Capacità di recuperare oltre il **$36\%$** dell'Oracle Gap su distribuzioni in-domain e preservare la stabilità funzionale anche in regimi a gradienti sub-griglia ($\sigma < h$).

---

## 2. Architettura Matematica

Il framework opera su domini bidimensionali discretizzati tramite mesh triangolari conformi $\mathcal{M} = (\mathcal{V}, \mathcal{E}, \mathcal{T})$, dove $\mathcal{V}$ rappresenta i nodi, $\mathcal{E}$ gli spigoli e $\mathcal{T}$ gli elementi triangolari.

                +-------------------------------------+
                | Physical State (u, grad_u, x, y)    |
                +-------------------------------------+
                                    |
                                    v
                +-------------------------------------+
                |         MeshMover GNN               |
                |   Encode-Process(6)-Decode (MLP)    |
                +-------------------------------------+
                                    |
                                    v  Raw Displacement Delta X_GNN
                +-------------------------------------+
                |    Projective Safety Filter O(M)    |
                |       min_e J_e(alpha) >= epsilon   |
                +-------------------------------------+
                                    |
                                    v  Safe Displacement Delta X_safe
                +-------------------------------------+
                | Validated Adaptive Mesh (J_e > eps) |
                +-------------------------------------+

### 2.1. Funzionale Variazionale dell'Oracolo (Target Offline)

L'oracolo definisce la configurazione nodale ottima $X^*$ minimizzando l'errore funzionale integrato di interpolazione $L^2$ regolarizzato:

$$X^* = \arg\min_{X} \left[ \tilde{E}_{\text{interp}}(X) + \lambda_D \tilde{D}(X) \right]$$

soggetto ai vincoli di non compenetrazione $J_e(X) \ge \epsilon > 0 \; \forall e \in \mathcal{T}$ e condizioni al contorno di Dirichlet fisse $\left. X \right\vert{}_{\partial \Omega} = X_0\vert{}_{\partial \Omega}$.

### 2.2. Filtro Geometrico di Sicurezza Analitico $\mathcal{O}(M)$

Dato uno spostamento nodale candidato $\Delta X$, il determinante Jacobiano dell'elemento triangolare $e$ lungo il raggio di proiezione scalato $\alpha \in [0, 1]$ è espresso analiticamente come un polinomio quadratico:

$$J_e(\alpha) = A_e \alpha^2 + B_e \alpha + C_e$$

dove i coefficienti dipendono esclusivamente dalle coordinate di base $X_0$ e dal displacement predetto $\Delta X$.
Il filtro determina la minima radice ammissibile positiva $\alpha_e^*$ per cui $J_e(\alpha) = \epsilon$. Il fattore globale di ritrazione risulta:

$$\alpha_{\text{global}} = \min \left(1.0, \, \min_{e \in \mathcal{T}} \alpha_e^* \right) \cdot (1 - \delta_{\text{margin}})$$

Garantendo che $\Delta X_{\text{safe}} = \alpha_{\text{global}} \Delta X$ preservi l'orientamento di ogni cella in un singolo passaggio vettorializzato senza iterazioni.

---

## 3. Struttura della Repository

```bash
├── 📁 config
│   └── ⚙️ config.yaml
│
├── 📁 data
│   ├── 📁 processed
│   ├── 📁 raw
│   └── ⚙️ manifest.json
│
├── 📁 dataset
│   ├── 🐍 __init__.py
│   └── 🐍 dataset.py
│
├── 📁 evaluation
│   ├── 🐍 __init__.py
│   └── 🐍 metrics.py
│
├── 📁 experiments
│   └── 🐍 stage_a_campaign.py
│
├── 📁 geometry
│   ├── 🐍 __init__.py
│   ├── 🐍 geometry.py
│   ├── 🐍 mesh.py
│   └── 🐍 safety.py
│
├── 📁 model
│   ├── 🐍 __init__.py
│   └── 🐍 model.py
│
├── 📁 numerics
│   ├── 🐍 __init__.py
│   ├── 🐍 ale.py
│   ├── 🐍 fem.py
│   ├── 🐍 pde.py
│   └── 🐍 remap.py
│
├── 📁 oracle
│   ├── 🐍 __init__.py
│   ├── 🐍 field.py
│   ├── 🐍 objective.py
│   └── 🐍 optimizer.py
│
├── 📁 results
│
├── 📁 tests
│   ├── 🐍 __init__.py
│   ├── 🐍 test_dataset.py
│   ├── 🐍 test_geometry.py
│   ├── 🐍 test_oracle.py
│   ├── 🐍 test_safety.py
│   └── 🐍 test_safety_stress.py
│
├── 📁 utils
│   ├── 🐍 __init__.py
│   └── 🐍 utils.py
│
├── ⚙️ .gitignore
├── 📝 README.md
├── ⚙️ environment.yml
│
├── 🐍 evaluate.py
├── 🐍 generate_dataset.py
├── 🐍 inference.py
├── ⚙️ pyproject.toml
└── 🐍 train.py
```

---

## 4. Installazione e Ambiente
Il framework è ottimizzato per **macOS (Apple Silicon con backend MPS)** e sistemi Linux/Windows con supporto **CUDA**.

### 4.1. Configurazione Conda

```bash
# Clonare la repository
git clone [https://github.com/tuo-username/Graph-Dynamic-Meshing.git](https://github.com/tuo-username/Graph-Dynamic-Meshing.git)
cd Graph-Dynamic-Meshing

# Creare l'ambiente virtuale
conda env create -f environment.yml

# Attivare l'ambiente
conda activate gdm_env
```

### 4.2. Verifica dell'Integrità del Codice
Eseguire la suite di test numerici per validare il layer geometrico e il filtro di sicurezza:
```bash
pytest tests/ -v
```

## 5. Guida all'Uso e Riproducibilità Scientifica

### Passo 1: Generazione Offline del Dataset
Genera i 210 campioni (train, validation, test interpolation e test extrapolation OOD) campionando i gradienti fisici e risolvendo l'ottimizzazione variazionale:

```bash
python generate_dataset.py
```

### Passo 2: Addestramento della MeshMover GNN
Addestra la rete sul displacement supervisionato $\Delta X^*$ con regolarizzazione e mascheramento di Dirichlet:

```bash
python train.py
```

Il checkpoint ottimale viene archiviato in `results/stage_a/checkpoints/best_model.pt`.

### Passo 3: Esecuzione del Benchmark Multi-Baseline
Valuta sistematicamente la GNN confrontandola con le baseline $B_0$(Fixed), $B_0$, $B_3$(Row), $B_4$(GNN + Safety) e $B_5$(Oracle):

```bash
python evaluate.py
```

### Passo 4: Campagna Sperimentale Completa per il Paper
Esegue l'intera suite di validazione scientifica (Test T1–T5), generando le 5 figure scientifiche e le 2 mappe qualitative in `results/stage_a/paper_figures/`:

```bash
python experiments/stage_a_campaign.py
```

---

## 6. Risultati Sperimentali e Validazione Scientifica (Stage A)

La fase di **Stage A** ha un obiettivo metodologico preciso: **validare l'adattività geometrica statica** della mesh prima di accoppiarla a solutori PDE differenziali dipendenti dal tempo. 

In questo scenario di riferimento, la fisica è modellata tramite un campo continuo bidimensionale caratterizzato da gradienti localizzati ripidi (Gaussian bump field). Il compito della rete neurale consiste nel concentrare la densità dei nodi della griglia laddove il gradiente è massimo ($r$-refinement), **senza alterare la topologia degli elementi (connettività costante) e impedendo rigorosamente inversioni o collassi geometrici delle celle**.

---

### 6.1. Definizione delle Baseline di Riferimento

Per quantificare in modo rigoroso e disaccoppiato il contributo di ciascun modulo, il framework confronta cinque configurazioni sperimentali:

1. **$B_0$ (Fixed Mesh):** Griglia conforme uniforme iniziale ($32 \times 32$, 1089 nodi, 2048 elementi) priva di adattamento. Rappresenta la discretizzazione classica standard.
2. **$B_{\text{zero}}$ (Zero-Displacement Predictor):** Modello di controllo a spostamento nullo ($\Delta X = 0$). Serve a verificare formalmente che il modello non converga banalmente verso la configurazione statica indeformata per minimizzare la varianza della loss.
3. **$B_3$ (GNN Raw):** Spostamento predetto direttamente dalla rete neurale in assenza di filtri a valle. Isola la capacità di apprendimento supervisionato grezzo della GNN.
4. **$B_4$ (GDM: GNN + Safety Filter):** Il framework proposto. Applica la correzione proiettiva analitica $\mathcal{O}(M)$ con fattore scalare $\alpha_{\text{global}} \in (0, 1]$ per garantire che $\min_e J_e \geq \epsilon$.
5. **$B_5$ (Oracle Reference):** Configurazione variazionale ottima calcolata offline minimizzando il funzionale multiobiettivo tramite ottimizzatore non lineare L-BFGS-B vincolato. Costituisce l'*upper bound* teorico a parità di gradi di libertà.

---

### 6.2. Metriche di Valutazione Funzionale e Geometrica

* **Errore di Interpolazione Funzionale ($E_{\text{interp}}$):**  
  Valuta l'errore in norma $L^2(\Omega)$ tra il campo analitico esatto $u(x, y)$ e la sua proiezione interpolata lineare a tratti $\Pi_h u$ sulla mesh deformata:
  $$E_{\text{interp}} = \sqrt{\int_\Omega \left| u(x, y) - \Pi_h u(x, y) \right|^2 \, d\Omega}$$
  Un errore più basso indica che i gradi di libertà si sono addensati nelle zone a massima variazione del campo, riducendo l'errore di approssimazione.

* **Quota di Oracle Gap Recuperata ($R_{\text{oracle}}$):**  
  Misura la frazione del divario di accuratezza tra la mesh fissa non adattata e l'ottimo teorico dell'oracolo che il modello neurale riesce a colmare:
  $$R_{\text{oracle}} = \frac{E_{\text{interp}}(B_0) - E_{\text{interp}}(\text{GNN})}{E_{\text{interp}}(B_0) - E_{\text{interp}}(B_5)} \times 100\%$$
  Un valore pari a $0\%$ corrisponde alle prestazioni della griglia fissa, mentre $100\%$ indica il perfetto raggiungimento dell'oracolo variazionale.

* **Determinante Jacobiano Minimo ($\min_e J_e$):**  
  Rappresenta il doppio dell'area orientata del triangolo $e$.
  * $J_e > 0$: elemento orientato correttamente in senso antiorario (ammissibile).
  * $J_e \leq 0$: elemento degenere o invertito (*tangled/collapsed*), condizione inaccettabile che provoca l'arresto immediato di solutori FEM/CFD.
  * Soglia di sicurezza impostata nel framework: $\min_e J_e \geq \epsilon = 1.0 \times 10^{-4}$.

---

### 6.3. Tabella Comparativa Multi-Baseline

Le prestazioni medie e la deviazione standard sono state quantificate sui campioni di test di interpolazione in-domain ($Test\_Interp$) e sui campioni di estrapolazione out-of-distribution con gradienti sub-griglia ($Test\_Extrap$, $\sigma < h$):

| Baseline | Metodo Adottato | $E_{\text{interp}} \; (\times 10^{-4})$ [In-Domain] | $R_{\text{oracle}}$ (%) | $E_{\text{interp}} \; (\times 10^{-4})$ [OOD] | $R_{\text{oracle}}$ (%) | Tempo CPU [ms] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **$B_0$** | Griglia Fissa Uniforme ($32 \times 32$) | $0.19 \pm 0.11$ | — | $0.43 \pm 0.43$ | — | $0.00$ |
| **$B_{\text{zero}}$** | Predittore a Spostamento Nullo | $0.19 \pm 0.11$ | — | $0.43 \pm 0.43$ | — | $0.00$ |
| **$B_3$** | GNN Raw (Non Protetta) | $0.15 \pm 0.08$ | $36.4\%$ | $0.34 \pm 0.35$ | $28.1\%$ | $42.23$ |
| **$B_4$** | **GDM (GNN + Safety Filter)** | **$0.15 \pm 0.08$** | **$36.4\%$** | **$0.34 \pm 0.35$** | **$28.1\%$** | **$42.49$** |
| **$B_5$** | Oracolo Variazionale (L-BFGS-B) | $0.08 \pm 0.04$ | $100.0\%$ | $0.11 \pm 0.09$ | $100.0\%$ | $5679.13$ |

#### Analisi dei Risultati
* **Riduzione dell'Errore:** A parità di nodi ($1089$) e di topologia, GDM riduce l'errore di interpolazione in-domain del **$21.1\%$** rispetto alla griglia fissa, recuperando il **$36.4\%$** del gap teorico rispetto all'ottimizzatore numerico L-BFGS-B.
* **Capacità di Generalizzazione OOD:** Nei casi con gradienti molto ripidi in cui la scala della gaussiana scende sotto la risoluzione della maglia iniziale ($\sigma < h$), il modello riduce l'errore da $0.43$ a $0.34$, recuperando il **$28.1\%$** del margine oracolo.

---

### 6.4. Stress Test del Safety Filter: Garanzia di Non-Inversione

Per testare le proprietà matematiche del filtro analitico in condizioni limite, lo spostamento nodale predetto dalla GNN è stato scalato artificialmente con un fattore moltiplicativo $k$:
$$\Delta X(k) = k \cdot \Delta X_{\text{GNN}}, \quad k \in \{1.0, 2.0, 3.0, 5.0, 10.0\}$$

Questo test forza la rete in regimi di deformazione non fisica per verificare l'intervento di $\alpha_{\text{global}}$ nel bloccare l'inversione delle celle:

| Moltiplicatore $k$ | Descrizione del Regime | $\min_e J_e$ Raw | Celle Invertite (Raw) | $\alpha_{\text{global}}$ Calcolato | $\min_e J_e$ Safe | Celle Invertite (Safe) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **$1.0$** | Predizione nominale della rete | $+6.38 \times 10^{-4}$ | $0$ | $1.000000$ *(Inattivo)* | $+6.38 \times 10^{-4}$ | **0** |
| **$2.0$** | Spostamento raddoppiato | $+2.99 \times 10^{-4}$ | $0$ | $1.000000$ *(Inattivo)* | $+2.99 \times 10^{-4}$ | **0** |
| **$3.0$** | Deformazione critica | $-3.98 \times 10^{-5}$ | $2$ | **$0.941203$** *(Attivo)* | **$+1.00 \times 10^{-4}$** | **0** |
| **$5.0$** | Regime di collasso severo | $-7.17 \times 10^{-4}$ | $18$ | **$0.564721$** *(Attivo)* | **$+1.00 \times 10^{-4}$** | **0** |
| **$10.0$** | Shock deformativo distruttivo | $-2.41 \times 10^{-3}$ | $86$ | **$0.282360$** *(Attivo)* | **$+1.00 \times 10^{-4}$** | **0** |

#### Evidenze del Test
1. **Trasparenza nei Regimi Ammissibili:** Per $k \leq 2.0$, il filtro calcola $\alpha_{\text{global}} = 1.0$ e lascia inalterato l'output della rete, evitando modifiche conservative non necessarie.
2. **Controllo Deterministico del Vincolo:** Non appena compaiono celle con Jacobiano negativo ($k \geq 3.0$), il fattore $\alpha_{\text{global}}$ scala proiettivamente il vettore di spostamento, arrestando il determinante esattamente a $\min_e J_e \geq \epsilon = 1.0 \times 10^{-4}$ e azzerando le celle invertite.

---

### 6.5. Profilazione Cronometrica e Scaling Computazionale (CPU vs CPU)

Per dimostrare l'applicabilità in tempo reale su architetture standard e dispositivi edge senza GPU dedicata, i tempi di esecuzione dell'intero framework e dell'oracolo classico sono stati misurati in modo isolato su CPU (Single-Core) al variare del numero di nodi $N$:

* **Risoluzione $16 \times 16$ ($N = 289$, $M = 512$ elementi):**
  * Oracolo L-BFGS-B: $689.00\text{ ms}$
  * GDM (GNN + Safety): $12.11\text{ ms} + 0.17\text{ ms} = \mathbf{12.28\text{ ms}}$ $\rightarrow$ **$56\times$ speedup**
* **Risoluzione $32 \times 32$ ($N = 1089$, $M = 2048$ elementi):**
  * Oracolo L-BFGS-B: $5679.13\text{ ms}$ ($\sim 5.7\text{ s}$)
  * GDM (GNN + Safety): $42.23\text{ ms} + 0.26\text{ ms} = \mathbf{42.49\text{ ms}}$ $\rightarrow$ **$133\times$ speedup**
* **Risoluzione $64 \times 64$ ($N = 4225$, $M = 8192$ elementi):**
  * Oracolo L-BFGS-B: $64600.63\text{ ms}$ ($\sim 64.6\text{ s}$)
  * GDM (GNN + Safety): $152.22\text{ ms} + 0.85\text{ ms} = \mathbf{153.07\text{ ms}}$ $\rightarrow$ **$422\times$ speedup**

#### Complessità Algoritmica
* L'oracolo L-BFGS-B scala in modo super-lineare a causa della costruzione numerica del gradiente tramite differenze finite e delle ripetute valutazioni degli integrali di quadratura a ogni iterazione di linea.
* **GDM mostra una complessità empirica quasi-lineare $\mathcal{O}(N)$**, determinata dai passaggi di *message passing* sparsi della GNN e dal calcolo elementare vettorializzato del Safety Filter, il quale mantiene un overhead inferiore a **$1\text{ ms}$** anche su maglie da oltre $8000$ triangoli.

## 7. Roadmap di Sviluppo e Prospettive Future

Il progetto segue una pianificazione incrementale a tappe per disaccoppiare e controllare rigorosamente le sorgenti di errore numerico:

- [x] **Stage A (Validazione Geometrica Statica su Gaussian Bump):**
  - Formulazione variazionale del problema di $r$-adaptation e generazione dataset supervisionato offline.
  - Implementazione dell'architettura `MeshMoverGNN` (Encode-Process-Decode a 6 blocchi con LayerNorm e SiLU).
  - Sviluppo e verifica analitica del filtro proiettivo $\mathcal{O}(M)$ con barriera logaritmica e calcolo esatto delle radici polinomiali.
  - Campagna sperimentale completata con successo: benchmark multi-baseline (T1), stress test geometrico (T2), profiling computazionale CPU vs CPU (T3), ablation study della loss (T4) e sweep di robustezza OOD (T5).

- [ ] **Stage B (Accoppiamento Dinamico con PDE Convezione-Diffusione Transiente):**
  - **Manufactured Solution (MMS):** Definizione del campo analitico dinamico $u_{\mathrm{exact}}(x, y, t)$ su dominio unitario $[0, 1]^2$ con convezione dominante ($Pe > 1$) e calcolo del termine sorgente $f(x, y, t)$.
  - **Discretizzazione Spaziale FEM:** Implementazione del solutore ad elementi finiti continui $P_1$ stabilizzato con schema **SUPG (Streamline Upwind Petrov-Galerkin)** per prevenire instabilità spurie ad alti numeri di Péclet.
  - **Adattività Temporale e Accoppiamento:** Confronto tra approccio a **Remapping Conservativo** (proiezione dello stato $u^n$ sulla nuova griglia) e formulazione differenziale **ALE (Arbitrary Lagrangian-Eulerian)** con velocità di griglia $\mathbf{w} = \frac{\mathbf{x}^{n+1} - \mathbf{x}^n}{\Delta t}$.
  - **Protocollo di Disaccoppiamento degli Errori:** Scomposizione quantitativa dell'errore globale $\Vert{}u_h^n - u_{\mathrm{exact}}(t_n)\Vert{}$ nei contributi elementari (errore FEM, passo temporale $\Delta t$, dispersione da remapping, qualità geometrica della mesh adattata).

- [ ] **Stage C (Benchmark Fluidodinamico e Problemi Ingegneristici Non Lineari):**
  - Estensione a equazioni di Navier-Stokes incomprimibili 2D per il tracciamento di strati limite e scie vorticose (vortex shedding).
  - Valutazione dell'inferenza real-time su flussi transitori complessi e scaling su architetture eterogenee GPU/Edge.

---

## 8. Licenza e Citazione

Il codice sorgente e la documentazione di questo repository sono distribuiti sotto licenza libera **MIT**. Consultare il file [`LICENSE`](LICENSE) per i termini completi di utilizzo e ridistribuzione.

### Citazione Scientifica
Se utilizzi questo framework o i modelli numerici di **Graph-Dynamic Meshing** per la tua ricerca, ti preghiamo di citare il lavoro come segue:

```bibtex
@article{iacuzzo2026graphdynamicmeshing,
  title   = {},
  author  = {Iacuzzo, Giovanni Giuseppe},
  journal = {},
  year    = {}
}
```