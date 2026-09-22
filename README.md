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
│   ├── 🐍 stage_a_campaign.py
│   ├── 🐍 stage_b_campaign.py
│   ├── 🐍 stage_b_diagnostics.py
│   └── 🐍 stage_b_final_ablations.py
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
│   ├── 🐍 dynamic_solver.py
│   ├── 🐍 fem.py
│   ├── 🐍 pde.py
│   ├── 🐍 remap.py
│   └── 🐍 remap_l2.py
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
│   ├── 🐍 test_dynamic_one_step.py
│   ├── 🐍 test_fem_convergence.py
│   ├── 🐍 test_fem_mms.py
│   ├── 🐍 test_geometry.py
│   ├── 🐍 test_oracle.py
│   ├── 🐍 test_pde_mms.py
│   ├── 🐍 test_remap.py
│   ├── 🐍 test_remap_diagnostics.py
│   ├── 🐍 test_remap_l2.py
│   ├── 🐍 test_safety.py
│   └── 🐍 test_safety_stress.py
│
├── 📁 utils
│   ├── 🐍 __init__.py
│   └── 🐍 utils.py
│
├── ⚙️ .gitignore
├── 📄 LICENSE
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

### 5.1. Pipeline dello Stage A (Adattività Geometrica Statica)

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

### 5.2. Pipeline dello Stage B (Accoppiamento Dinamico PDE Transiente)
### Passo 5: Test Unitari e di Consistenza Numerica
Verifica in modo disaccoppiato l'esattezza della soluzione manifatturata (MMS), la convergenza spaziale e temporale del solutore FEM e l'integrazione del singolo passo accoppiato:

```bash
# Verifica consistenza Manufactured Solution e forcing analitico
pytest tests/test_pde_mms.py -v

# Verifica ordini asintotici di convergenza spazio-temporale (O(h^2) e O(dt))
pytest tests/test_fem_convergence.py -v

# Verifica one-step coupling (FEM -> Feature Extraction -> GNN -> Safety -> Remap)
pytest tests/test_dynamic_one_step.py -v
```

### Passo 6: Suite Diagnostica Operatori di Remapping
Esegue l'analisi quantitativa comparativa tra l'interpolazione continua pointwise $P_1$ e l'operatore di proiezione variazionale $L_2$-Projection:

```bash
# Diagnostica P1: Esattezza (R1), Convergenza (R2), Ampiezza (R3), Localizzazione (R4), Accumulo (R5)
pytest tests/test_remap_diagnostics.py -s -v

# Diagnostica L2-Projection: Esattezza (L1), Convergenza (L2), Ampiezza (L3), Omotopia (L4), Costo (L5), Massa (L6)
pytest tests/test_remap_l2.py -s -v

# Batteria diagnostica dinamica D1-D5 (Remap isolato, FEM discolpato D2, Reset esatto D3, Sweep frequenza D5)
python experiments/stage_b_diagnostics.py
```

### Passo 7: Campagna Sperimentale Dinamica Multi-Modello
Esegue la simulazione transiente completa a 100 time-step sulle 4 configurazioni ($B_0$Fixed, $B_1$GDM w/ $P_1$, $B_2$ GDM w/ $L_2$, $B_3$ Instantaneous Interpolation Oracle), generando la tabella riassuntiva LaTeX e le 8 figure scientifiche per il paper:

```bash
python experiments/stage_b_campaign.py
```

I grafici e i report vengono salvati rispettivamente in `results/stage_b/figures/` e `results/stage_b/tables/table_stage_b_summary.tex`.

### Passo 8: Studi Ablativi Conclusivi dello Stage B
Esegue l'ablation della sorgente del gradiente nodale (FEM vs Esatto MMS) e lo sweep di sensibilità alla frequenza di adattamento temporale ($\Delta n_{adapt} \in \{5,10,20,50\}$):

```bash
python experiments/stage_b_final_ablations.py
```

I dati strutturati vengono esportati in `results/stage_b/ablations/stage_b_final_ablations.json`.

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

---

## 7. Risultati Sperimentali e Validazione Dinamica (Stage B)

La fase di **Stage B** ha esteso il framework GDM dall'adattività statica all'**accoppiamento dinamico transiente con solutori differenziali alle derivate parziali (PDE)**. L'obiettivo primario consiste nel verificare se la mesh deformata in tempo reale dalla GNN mantenga stabilità geometrica, accuratezza formale e consistenza fisica lungo una traiettoria temporale finita senza degradare la discretizzazione numerica.

Il problema test di riferimento è l'equazione di **Convezione-Diffusione instazionaria 2D** su dominio $\Omega = [0, 1]^2$:
$$\frac{\partial u}{\partial t} + \mathbf{b} \cdot \nabla u - \nu \nabla^2 u = f(x, y, t), \quad t \in [0, T]$$
con velocità di convezione $\mathbf{b} = [0.4, 0.2]^T$, diffusività $\nu = 10^{-3}$, $T = 0.5$ e $\Delta t = 0.005$ ($100$ time-step). Per condurre una verifica di consistenza esatta a livello analitico, la PDE è integrata con il **Metodo delle Soluzioni Costruite (MMS)**, adottando un fronte gaussiano convettivo smorzato:
$$u_{\mathrm{exact}}(x, y, t) = \exp(-\beta t) \exp\left( -\frac{(x - x_c(t))^2 + (y - y_c(t))^2}{2\sigma^2} \right)$$
con $\sigma = 0.06$, $\beta = 0.5$ e centro mobile $\mathbf{x}_c(t) = \mathbf{x}_0 + \mathbf{b}t$.

---

### 7.1. Suite Diagnostica Metodologica (D1–D5 e R1–R5)

I test preliminari con remapping standard lineare pointwise $P_1$ su mesh $32 \times 32$ evidenziavano un degrado di accuratezza dinamica (GDM a $12.72\%$ contro il $9.03\%$ della mesh fissa). Per isolare il contributo di ciascun sottosistema senza ricorrere a un riaddestramento cieco della GNN, è stata eseguita una campagna diagnostica disaccoppiata:

* **Verifica del Solutore FEM (D2):** Valutando un singolo passo temporale $\Delta t$ a partire dalla soluzione esatta $u(t_n)$, la discretizzazione FEM $P_1$ con Backward Euler produce:
  * Errore $L^2$ su Fixed Mesh: **$4.003\%$**
  * Errore $L^2$ su GNN Mesh: **$3.674\%$**  
  * *Esito:* Il solutore FEM e la qualità geometrica dei nodi generati dalla GNN sono discolpati; a parità di stato iniziale, la mesh addensata dalla GNN riduce l'errore del passo PDE.
* **Isolamento della Dissipazione di Trasferimento (D3):** Eseguendo la simulazione dinamica a 100 passi con azzeramento analitico dell'errore di remapping ($u(X_{\text{new}}) = u_{\text{exact}}$ ad ogni riadattamento), l'errore dinamico finale scende da **$12.72\%$** a **$3.85\%$** (contro il $9.03\%$ della griglia fissa). Questo ha provato in modo inconfutabile che il decadimento era interamente imputabile all'accumulo di dispersione numerica nel trasferimento di stato.
* **Analisi di Convergenza e Accumulo del Remapping (R1–R5):**
  * **R1 (Esattezza):** Conservazione di costanti ($u=c$) e polinomi lineari ($u=ax+by+c$) a precisione di macchina ($\sim 10^{-16}$).
  * **R2 (Convergenza Spaziale):** L'operatore $P_1$ pointwise scala come $\mathcal{O}(h^2)$ asintotico ($p = 1.96 \to 2.01$), ma a risoluzione $32 \times 32$ presenta un errore assoluto per step del **$2.46\%$**.
  * **R5 / L4 (Accumulo Omotopico a Target Fisso):** Su 10 trasferimenti progressivi concatenati verso la stessa mesh target, l'accumulo di proiezioni $P_1$ reiterate amplifica l'errore a **$5.95\%$** per via del filtraggio passa-basso della combinazione convessa.

---

### 7.2. Operatore di Trasferimento di Stato: $L^2$-Projection

Per eliminare l'accumulo dissipativo senza ricorrere a un'ottimizzazione variazionale runtime, è stato formulato e implementato un operatore variazionale globale basato su **$L^2$-Projection continua**:
$$\int_\Omega \phi_i^{\text{new}} u^{\text{new}} \, d\Omega = \int_\Omega \phi_i^{\text{new}} u^{\text{old}} \, d\Omega \quad \iff \quad M^{\text{new}} \mathbf{u}^{\text{new}} = \mathbf{b}$$
dove il termine di carico continuo $\mathbf{b}$ è valutato integrando il campo d'origine $u^{\text{old}}$ tramite quadratura di Dunavant di grado 5 (7 punti per cella).

* **Riduzione dell'Errore di Trasferimento (Mesh $32 \times 32$):** L'errore $L^2$ del singolo riadattamento scende da $2.46\%$ ($P_1$) a **$0.17\%$** ($L^2$-projection), con una riduzione dell'errore di oltre **$14.5\times$**.
* **Conservazione dell'Integrale di Massa (L6):** L'identità variazionale $\mathbf{1}^T M^{\text{new}} \mathbf{u}^{\text{new}} = \mathbf{1}^T \mathbf{b}$ è soddisfatta a precisione macchina ($< 10^{-12}$), con uno scarto di massa discreta fisica confinato allo $0.02\%$.
* **Accumulo Consecutivo (10 trasferimenti):** L'errore cumulato si arresta ad appena lo **$0.332\%$** (contro il $5.949\%$ di $P_1$).

---

### 7.3. Benchmark Dinamico Comparativo Multi-Modello

Il benchmark finale ha confrontato su $100$ time-step le quattro configurazioni operative a partire dal medesimo stato iniziale su maglia $32 \times 32$ ($1089$ nodi, $2048$ elementi):

1. **$B_0$ (Fixed Mesh):** Griglia uniforme stazionaria priva di adattamento.
2. **$B_1$ (GDM w/ $P_1$ Remap):** Riadattamento con la rete frozen di Stage A e remapping baricentrico pointwise (baseline iniziale).
3. **$B_2$ (GDM w/ $L^2$-Projection):** Framework proposto nominale (GNN + Safety Filter + proiezione $L^2$).
4. **$B_3$ (Instantaneous Interpolation Oracle w/ $L^2$-Projection):** Riadattamento guidato dall'ottimizzatore variazionale offline di Stage A ricalcolato ad ogni intervallo temporale.

| Configurazione | Operatore Remap | Errore $L^2(T)$ Finale | Errore $L^2$ Time-Integrated | $\min_{t,e} J_e$ | Tempo FEM [s] | Tempo Remap [s] | Tempo Totale [s] | Speedup vs Oracle |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$B_0$ (Fixed Mesh)** | Nessuno | $9.03\%$ | $7.96\%$ | $9.77 \times 10^{-4}$ | $1.08$ | $0.00$ | **$1.08$** | — |
| **$B_1$ (GDM Nominal $P_1$)** | $P_1$ Pointwise | $12.72\%$ | $10.15\%$ | $6.59 \times 10^{-4}$ | $1.15$ | $0.27$ | **$2.58$** | $\mathbf{31.0\times}$ |
| **$B_2$ (GDM Proposto $L^2$)** | $L^2$-Projection | **$8.99\%$** | **$7.89\%$** | $6.59 \times 10^{-4}$ | $1.16$ | $3.53$ | **$5.25$** | $\mathbf{15.2\times}$ |
| **$B_3$ (Interp. Oracle $L^2$)** | $L^2$-Projection | $9.57\%$ | $8.24\%$ | $2.69 \times 10^{-4}$ | $1.15$ | $3.55$ | **$79.93$** | $1.0\times$ *(Ref)* |


#### Risultanze Scientifiche
1. **Recupero della Precisione PDE:** L'adozione del remapper $L^2$ annulla integralmente il degrado introdotto dal remapping $P_1$ ($12.72\% \to 8.99\%$), attestando la simulazione su livelli di accuratezza comparabili e leggermente superiori rispetto alla griglia fissa non deformata ($9.03\%$).
2. **Natura dell'Instantaneous Interpolation Oracle:** L'oracolo variazionale ($B_3$), pur addensando i nodi in modo estremo sul picco del campo, ottiene un errore dinamico finale peggiore ($9.57\%$). Questo dimostra che ottimizzare l'interpolazione puramente geometrica a istanti discreti non coincide con la minimizzazione dell'errore globale di una PDE differenziale nel tempo.
3. **Garanzia Deterministica di Sicurezza:** In nessuna configurazione dinamica si registrano elementi degeneri o invertiti ($\min_{t,e} J_e \geq 6.59 \times 10^{-4} > \epsilon$). Il Safety Filter garantisce stabilità incondizionata lungo l'intera traiettoria.
4. **Vantaggio Computazionale Real-Time:** GDM ($B_2$) porta a termine i 100 time-step completi con 10 riadattamenti e proiezioni $L^2$ globali in **$5.25\text{ s}$**, preservando un'accelerazione di **$15.2\times$** rispetto all'approccio basato sull'oracolo variazionale classico ($79.93\text{ s}$).

---

### 7.4. Ablation Studies Conclusivi dello Stage B

* **Sorgente del Gradiente (Ablation $B_1$ vs $B_2$):**  
  Confrontando l'alimentazione della GNN con il gradiente numerico nodale ricavato a runtime dal FEM rispetto al gradiente analitico esatto della MMS:
  * Errore finale $L^2$ con Gradiente FEM: **$8.989\%$** (Time-Int: $7.890\%$)
  * Errore finale $L^2$ con Gradiente Esatto MMS: **$8.979\%$** (Time-Int: $7.889\%$)  
  Lo scarto tra le due configurazioni è di appena **$0.010$ punti percentuali**, dimostrando che il framework opera in piena autonomia da soluzioni analitiche senza subire degradazioni dovute all'estrazione numerica delle feature.

* **Frequenza di Adattamento (Sweep dell'Intervallo su GDM + $L^2$):**

| Intervallo $\Delta n_{\text{adapt}}$ | N. Riadattamenti | Errore Finale $L^2$ | Errore Time-Int $L^2$ | Tempo Remap [s] | Tempo Totale [s] |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$5$** | $20$ | $8.984\%$ | $7.819\%$ | $7.10$ | $9.50$ |
| **$10$** *(Nominale)* | $10$ | $8.989\%$ | $7.890\%$ | $3.53$ | $5.25$ |
| **$20$** | $5$ | $9.001\%$ | $7.932\%$ | $1.83$ | $3.11$ |
| **$50$** | $2$ | $9.039\%$ | $7.957\%$ | $0.70$ | $1.82$ |

All'aumentare della frequenza di deformazione l'errore temporale integrato cala progressivamente a fronte di un costo computazionale quasi linearmente proporzionale al numero di proiezioni $L^2$. La configurazione **$\Delta n_{\text{adapt}} = 10$** si conferma il punto operativo ottimale, garantendo il recupero dell'accuratezza PDE con un tempo totale di calcolo limitato a $\sim 5\text{ s}$.

---

## 8. Roadmap di Sviluppo e Prospettive Future

Il progetto segue una pianificazione incrementale a tappe per disaccoppiare e controllare rigorosamente le sorgenti di errore numerico:

- [x] **Stage A (Validazione Geometrica Statica su Gaussian Bump):**
  - Formulazione variazionale del problema di $r$-adaptation e generazione dataset supervisionato offline.
  - Implementazione dell'architettura `MeshMoverGNN` (Encode-Process-Decode a 6 blocchi con LayerNorm e SiLU).
  - Sviluppo e verifica analitica del filtro proiettivo $\mathcal{O}(M)$ con barriera logaritmica e calcolo esatto delle radici polinomiali.
  - Campagna sperimentale completata con successo: benchmark multi-baseline (T1), stress test geometrico (T2), profiling computazionale CPU vs CPU (T3), ablation study della loss (T4) e sweep di robustezza OOD (T5).

- [x] **Stage B (Accoppiamento Dinamico con PDE Convezione-Diffusione Transiente):**
  - Implementazione Manufactured Solution (MMS) bidimensionale con convezione ad alta dinamica ($Pe > 1$) e verifica ordini asintotici di convergenza ($L^2 \to 2.0$, $H^1 \to 1.0$, $\Delta t \to 1.0$).
  - Integrazione completa del loop temporale accoppiato (FEM Backward Euler + GNN Inference + Deterministic Safety Filter + State Remapper).
  - Isolamento quantitativo della dissipazione numerica di proiezione baricentrica tramite suite diagnostica formale (D1–D5, R1–R5).
  - Implementazione e validazione dell'operatore di remapping a **$L^2$-Projection continua**, con abbattimento dell'errore di trasferimento di oltre $14.5\times$ e conservazione globale della massa.
  - Chiusura della campagna dinamica: validazione dell'indipendenza dal gradiente esatto ($\Delta \approx 0.01\%$), quantificazione del trade-off frequenza/accuratezza e conseguimento di un'accelerazione di **$15.2\times$** sull'Oracolo variazionale.

- [ ] **Stage C (Benchmark Fluidodinamico e Problemi Ingegneristici Non Lineari):**
  - Estensione a equazioni di Navier-Stokes incomprimibili 2D per il tracciamento di strati limite e scie vorticose (vortex shedding).
  - Valutazione dell'inferenza real-time su flussi transitori complessi e scaling su architetture eterogenee GPU/Edge.

---

## 9. Licenza e Citazione

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