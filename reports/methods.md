# Methods

The mathematics of every step, in pipeline order. Each formula names the function that
implements it (module `ehrgraph.<module>`), and every parameter is in `config/config.yaml`; the
values given are the project's. Results are in `findings.md`, `results.md` and
`embedding_results.md`.

## 0. Notation

| Symbol | Meaning |
| --- | --- |
| $i, j$ | patients (graph nodes), $i, j \in \{1, \dots, N\}$; $N = 2{,}581$ in the model cohort |
| $c, a, b$ | condition codes (SNOMED-CT), $c \in \{1, \dots, C\}$; $C = 204$ feature codes |
| $t, s$ | yearly windows; window $t$ is the calendar year $Y_t$, the interval $[Y_t\text{-}01\text{-}01,\ (Y_t{+}1)\text{-}01\text{-}01)$ |
| $T$ | number of feature windows: 2017-2022, so $T = 6$ |
| $\tau$ | the prediction cutoff, 2023-01-01; $H = 3$ years is the label horizon $[\tau, \tau + H)$ |
| $\mathbf{1}[\cdot]$ | indicator: 1 if the condition holds, else 0 |
| $\lVert x \rVert_2$ | Euclidean norm; $\operatorname{norm}(x) = x / \lVert x \rVert_2$ (and $0$ for $x = 0$) |
| $\sigma(z)$ | logistic function $1 / (1 + e^{-z})$ |
| $I$ | identity matrix; $\operatorname{diag}(v)$ is the diagonal matrix with $v$ on its diagonal |
| $\odot$ | element-wise product |

Bold capitals are matrices, rows are patients. Every $N \times N$ matrix is stored sparse
(`scipy.sparse`); none is ever formed dense.

## 1. Clinical events

### 1.1 Code merges and the clinical code filter (`data.merge_codes`, `data.filter_clinical_events`)

A condition event is a row of `conditions.csv`: patient $i$, code $c$, an onset date $d$ (the
START of its encounter, UTC) and an optional stop date. Codes that record one concept twice are
first renamed to one canonical code (`data.code_merge`). Then a code $c$ with the set of
descriptions $\mathcal{D}(c)$ (Synthea versions word some codes differently) is **kept** if

$$
\exists\, w \in \mathcal{D}(c):\quad \operatorname{lower}(w) \text{ ends with a suffix in } \mathcal{S}
\ \ \text{or}\ \ \operatorname{lower}(w) \in \operatorname{lower}(\mathcal{W}),
$$

with $\mathcal{S} = \{\text{"(disorder)"}\}$ and $\mathcal{W}$ a reviewed whitelist of 19
clinical findings (prediabetes, obesity, chronic pain, substance use, cardiac history, ...).
Social, administrative and symptom codes are dropped: 223 of 296 codes and 40,957 of 106,926
events are kept.

### 1.2 Active-condition windows (`data.active_until`, `data.active_in_year`)

Let $d_e$ be the onset of event $e$, $\mathrm{stop}_e$ its STOP date (possibly missing) and
$\delta_i$ the death date of its patient (possibly missing). The event is active until

$$
u_e = \min(\mathrm{stop}_e,\ \delta_i), \qquad \text{missing values ignored; } u_e = \infty \text{ if both are missing,}
$$

because Synthea leaves conditions open at death. The event is **active in window $t$** if

$$
d_e < (Y_t{+}1)\text{-}01\text{-}01 \quad\text{and}\quad u_e \ge Y_t\text{-}01\text{-}01 .
$$

The binary window matrix is $\mathbf{B}_t \in \{0,1\}^{N \times C}$ with
$B_{t,ic} = 1$ if any event of code $c$ of patient $i$ is active in window $t$. An episode is
**chronic** if it has no STOP date or lasts more than 365 days (`features.chronic_mask`).

Windows run 2017-2025. Synthea's export keeps a full record only for the 10 years before its
run date, so earlier years hold mostly conditions still open then (clinical events dated per
year: about 400 before 2017, about 2,500 from 2017); the windows start at the first full year.

### 1.3 Redundant general codes (`features.co_recorded_codes`)

Let $E(c)$ be the set of encounters that record code $c$. Code $s$ is a **specific form** of
code $g$ if

$$
\frac{|E(s) \cap E(g)|}{|E(s)|} \ \ge\ \rho \quad\text{and}\quad |E(s)| \le |E(g)|, \qquad \rho = 0.95 .
$$

The **coverage** of $g$ is the share of its encounters that also record one of its specific
forms, $\big|E(g) \cap \bigcup_{s} E(s)\big| / |E(g)|$. A code with coverage $\ge \rho$ is a
redundant general code (e.g. "Fracture of bone", always recorded with the specific fracture);
after clinical review, 10 such codes (`features.redundant_codes`) are dropped from the feature
vocabulary, leaving $C = 204$. The all-pairs comorbidity table keeps them.

## 2. Features

### 2.1 TF-IDF (`features.fit_idf`, `features.tfidf_transform`)

Each non-empty patient-window row of $\mathbf{B}_1, \dots, \mathbf{B}_T$ is a document. With
$n$ documents, of which $\mathrm{df}_c$ contain code $c$, the smoothed inverse document
frequency is

$$
\mathrm{idf}_c = \ln\frac{1 + n}{1 + \mathrm{df}_c} + 1 .
$$

It is fitted on the feature windows only (2017-2022) and saved. Each window is weighted and its
rows L2-normalized:

$$
\mathbf{X}_t = \operatorname{rownorm}\!\big(\mathbf{B}_t \operatorname{diag}(\mathrm{idf})\big).
$$

Rare codes weigh more than codes almost everyone has (gingivitis, viral sinusitis).

### 2.2 Cumulative time decay (`features.time_decay_cumulative`)

A condition seen in an earlier window still says something about the patient, with a weight
that decays with its age:

$$
\mathbf{X}^{\mathrm{cum}}_t = \sum_{s \le t} e^{-\lambda (Y_t - Y_s)}\, \mathbf{X}_s
= e^{-\lambda (Y_t - Y_{t-1})}\, \mathbf{X}^{\mathrm{cum}}_{t-1} + \mathbf{X}_t,
\qquad \lambda = 0.1\ \text{per year}.
$$

### 2.3 Per-window transition features (`features.patient_transition_matrix`, `features.transition_features`)

Each code enters a patient's trajectory once, at its first onset. A **transition** $a \to b$
of patient $i$ means $b$ was the first new condition after $a$'s onset (codes with the same
onset time share a step). For window $t$, only transitions that end before
$\min\big((Y_t{+}1)\text{-}01\text{-}01,\ \tau\big)$ are used, so no window sees a later
transition. With $n_{i,ab}^{(t)}$ the number of such transitions of patient $i$, patient $i$'s
empirical Markov transition matrix is

$$
P^{(t)}_{i,ab} = \frac{n^{(t)}_{i,ab}}{\sum_{b'} n^{(t)}_{i,ab'}} \quad (\text{0 if } a \text{ has no transition}),
$$

flattened into row $i$ of $\mathbf{M}_t \in \mathbb{R}^{N \times C^2}$. A truncated SVD is
fitted once, on the last feature window's block,
$\mathbf{M}_T \approx \mathbf{U}_r \boldsymbol{\Sigma}_r \mathbf{V}_r^\top$ with $r = 32$
components, and every window is projected on it: $\mathbf{S}_t = \mathbf{M}_t \mathbf{V}_r$.

### 2.4 The feature vector

Each block is row-normalized, the transition block is weighted by $w = 1$, and the result is
row-normalized again (`features.combine_features`):

$$
\mathbf{F}_t = \operatorname{rownorm}\Big[\ \operatorname{rownorm}(\mathbf{X}^{\mathrm{cum}}_t)\ \Big|\ w\, \operatorname{rownorm}(\mathbf{S}_t)\ \Big]
\in \mathbb{R}^{N \times (C + r)} = \mathbb{R}^{N \times 236}.
$$

The model cohort is the patients with a non-zero row in at least one feature window.

## 3. Comorbidity: lift and Jaccard (`features.comorbidity_counts`, `features.comorbidity_lift`, `features.top_comorbidity_pairs`)

Stack the non-empty binary rows into $\mathbf{Z}$ (units: patient-windows; for the chronic
table, one row per patient with the codes chronic-active in any feature window). The
co-occurrence counts and the population $P$ (number of units) give

$$
\mathbf{K} = \mathbf{Z}^\top \mathbf{Z}, \qquad K_{ab} = \#\{\text{units with } a \text{ and } b\}, \qquad K_{aa} = \#\{\text{units with } a\},
$$

$$
\operatorname{lift}(a, b) = \frac{K_{ab}}{K_{aa} K_{bb} / P} = \frac{\Pr(a, b)}{\Pr(a) \Pr(b)}, \qquad
J(a, b) = \frac{K_{ab}}{K_{aa} + K_{bb} - K_{ab}} .
$$

Lift is 1 for independent codes (it equals $e^{\mathrm{PMI}}$); the Jaccard index is the share
of units with $a$ or $b$ that have both. The heatmap orders the 30 most common chronic codes by
average-linkage clustering (Euclidean, optimal leaf order) of their profiles
$\big(\operatorname{clip}(\log_2 \operatorname{lift}(a, b), -3, 3)\big)_b$, with pairs never seen
together at $-3$ (`features.lift_profiles`, `features.lift_linkage`).

## 4. Patient-similarity graphs

### 4.1 k-NN graph (`graph.knn_graph`)

For the non-zero rows of $\mathbf{F}_t$, the cosine similarity is

$$
\operatorname{sim}(i, j) = \frac{\mathbf{f}_i^\top \mathbf{f}_j}{\lVert \mathbf{f}_i \rVert_2 \lVert \mathbf{f}_j \rVert_2} \in [0, 1]
$$

(features are non-negative). Let $\mathcal{N}_k(i)$ be the $k = 10$ most similar other patients.
The directed and the symmetrized weights are

$$
\tilde W_{ij} = \operatorname{sim}(i, j)\,\mathbf{1}[j \in \mathcal{N}_k(i)], \qquad
W_{ij} = \max(\tilde W_{ij}, \tilde W_{ji}),
$$

with non-positive weights dropped and no self-loops. A patient with an all-zero row (nothing
active yet) is left out of the search: an isolated node. The search computes distances a block
of rows at a time; no $N \times N$ matrix is built.

### 4.2 Normalized adjacency and Laplacian (`graph.normalize_adjacency`, `graph.laplacian`)

$$
\hat{\mathbf{A}}_t = \tilde{\mathbf{D}}^{-1/2} (\mathbf{W}_t + I)\, \tilde{\mathbf{D}}^{-1/2}, \qquad \tilde D_{ii} = 1 + \sum_j W_{t,ij},
$$

$$
\mathbf{L}_t = I - \mathbf{D}^{-1/2} \mathbf{W}_t \mathbf{D}^{-1/2}, \qquad D_{ii} = \sum_j W_{t,ij}\ \ (D_{ii}^{-1/2} := 0 \text{ if } D_{ii} = 0).
$$

$\hat{\mathbf{A}}$ (with self-loops, the GCN propagation matrix) has eigenvalues in $[-1, 1]$.
$\mathbf{L}$ is symmetric positive semi-definite with eigenvalues $0 = \lambda_1 \le \lambda_2
\le \dots \le 2$, one zero per connected component with an edge. For any $\mathbf{z} \in \mathbb{R}^N$,

$$
\mathbf{z}^\top \mathbf{L}\, \mathbf{z} = \tfrac12 \sum_{i,j} W_{ij} \Big(\tfrac{z_i}{\sqrt{D_{ii}}} - \tfrac{z_j}{\sqrt{D_{jj}}}\Big)^2,
$$

so eigenvectors with small eigenvalues vary little across strong edges.

### 4.3 Spectral embedding, eigengap and phenotypes (`graph.smallest_eigenpairs`, `graph.spectral_embedding`, `graph.eigengap`, `graph.spectral_clusters`)

On the largest connected component of the 2022 graph (all 2,581 patients), the smallest
eigenpairs $\mathbf{L}\mathbf{u}_m = \lambda_m \mathbf{u}_m$ are computed with Lanczos iteration
(`scipy.sparse.linalg.eigsh`, seeded start vector, sign fixed so each eigenvector's largest
entry is positive). Skipping the trivial eigenvectors ($\lambda = 0$), the **spectral
embedding** of patient $i$ is

$$
\mathbf{e}_i = \big(u_{2,i}, u_{3,i}, \dots, u_{d+1,i}\big), \qquad d = 16 \text{ (Phase 4)},\ 32 \text{ (Phase 7b)}.
$$

**Eigengap.** With $g_m = \lambda_{m+1} - \lambda_m$ over the first 20 eigenvalues, the number
of clusters is

$$
\hat k = \arg\max_{m \ge 2}\ g_m = 7 .
$$

**Phenotypes.** The rows of $[\mathbf{u}_1, \dots, \mathbf{u}_{\hat k}]$ are L2-normalized
(Ng-Jordan-Weiss) and clustered by k-means ($\hat k = 7$, 10 restarts). Each cluster is described
by its chronic conditions' prevalence ratio, cluster prevalence over overall prevalence.

**Graph drift** (`graph.neighbourhood_jaccard`): for consecutive windows,
$\big|\mathcal{N}_t(i) \cap \mathcal{N}_{t+1}(i)\big| / \big|\mathcal{N}_t(i) \cup \mathcal{N}_{t+1}(i)\big|$
over the patients with neighbours in both, where $\mathcal{N}_t(i)$ is $i$'s neighbour set in
$\mathbf{W}_t$.

## 5. The temporal graph convolutional network (`models.TemporalGCN`, `models.OnsetModel`)

### 5.1 GCN layer and spatial encoder

A graph convolution aggregates each patient's normalized neighbourhood, self included:

$$
\operatorname{GCN}(\mathbf{H}; \hat{\mathbf{A}}) = \hat{\mathbf{A}}\, \mathbf{H}\, \boldsymbol{\Theta} + \mathbf{1}\mathbf{b}^\top,
$$

with weights $\boldsymbol{\Theta}$ (Xavier-uniform initialization) and bias $\mathbf{b}$. The
spatial encoder, **shared by every window**, is two layers:

$$
\mathbf{Z}_t = \hat{\mathbf{A}}_t\, \operatorname{dropout}_p\!\Big(\operatorname{ReLU}\big(\hat{\mathbf{A}}_t \mathbf{F}_t \boldsymbol{\Theta}^{(1)} + \mathbf{1}\mathbf{b}^{(1)\top}\big)\Big)\, \boldsymbol{\Theta}^{(2)} + \mathbf{1}\mathbf{b}^{(2)\top} \in \mathbb{R}^{N \times d},
$$

$\boldsymbol{\Theta}^{(1)} \in \mathbb{R}^{236 \times 64}$, $\boldsymbol{\Theta}^{(2)} \in \mathbb{R}^{64 \times 32}$,
$d = 32$, dropout rate $p$. Row $\mathbf{z}_{i,t}$ is patient $i$'s embedding in year $t$: the
time series of embeddings is $(\mathbf{z}_{i,1}, \dots, \mathbf{z}_{i,T})$.

### 5.2 GRU update

A gated recurrent unit reads each patient's sequence, $\mathbf{h}_{i,0} = \mathbf{0}$:

$$
\begin{aligned}
\mathbf{r}_t &= \sigma(\mathbf{W}_{r} \mathbf{z}_t + \mathbf{b}_{r} + \mathbf{U}_{r} \mathbf{h}_{t-1} + \mathbf{c}_{r}) &&\text{reset gate}\\
\mathbf{g}_t &= \sigma(\mathbf{W}_{g} \mathbf{z}_t + \mathbf{b}_{g} + \mathbf{U}_{g} \mathbf{h}_{t-1} + \mathbf{c}_{g}) &&\text{update gate}\\
\mathbf{n}_t &= \tanh\!\big(\mathbf{W}_{n} \mathbf{z}_t + \mathbf{b}_{n} + \mathbf{r}_t \odot (\mathbf{U}_{n} \mathbf{h}_{t-1} + \mathbf{c}_{n})\big) &&\text{candidate state}\\
\mathbf{h}_t &= (1 - \mathbf{g}_t) \odot \mathbf{n}_t + \mathbf{g}_t \odot \mathbf{h}_{t-1}
\end{aligned}
$$

(patient index $i$ dropped), with input weights $\mathbf{W}_\cdot \in \mathbb{R}^{32 \times 32}$,
recurrent weights $\mathbf{U}_\cdot \in \mathbb{R}^{32 \times 32}$ and biases
$\mathbf{b}_\cdot, \mathbf{c}_\cdot$. $\mathbf{h}_{i,t}$ is the state after year $t$ and
$\mathbf{h}_i = \mathbf{h}_{i,T}$ the **trajectory embedding**.

### 5.3 Prediction heads

With demographics, $\mathbf{v}_i = [\mathbf{h}_i \mid \tilde a_i, s_i]$, where $\tilde a_i$ is the
age at $\tau$ standardized with training-split mean and standard deviation and $s_i = 1$ for
male; age and sex never enter the graphs or $\mathbf{F}_t$. Two linear heads give

$$
\ell_i = \mathbf{w}^\top \mathbf{v}_i + \beta \quad (\texttt{any\_onset}), \qquad
\boldsymbol{\ell}^{\mathrm{grp}}_i = \mathbf{W}^{\mathrm{grp}} \mathbf{v}_i + \boldsymbol{\beta}^{\mathrm{grp}} \in \mathbb{R}^{G} \quad (G = 9 \text{ onset groups}).
$$

### 5.4 Supervised loss (`models.masked_bce`, `models.onset_loss`)

Labels are $y \in \{0, 1\}$ or masked. Let $\mathcal{O}$ be the observed entries **of
training-split patients** and $\pi = \#\{y = 0\} / \#\{y = 1\}$ on them (per column for the
groups). The masked, positive-weighted binary cross-entropy is

$$
\mathcal{L}_{\mathrm{BCE}}(\ell, y; \pi) = \frac{1}{|\mathcal{O}|} \sum_{(i, g) \in \mathcal{O}} -\Big[\pi_g\, y_{ig} \log \sigma(\ell_{ig}) + (1 - y_{ig}) \log\big(1 - \sigma(\ell_{ig})\big)\Big],
$$

and the training loss is

$$
\mathcal{L} = \mathcal{L}_{\mathrm{BCE}}(\ell, y^{\mathrm{any}}; \pi^{\mathrm{any}}) + \gamma\, \mathcal{L}_{\mathrm{BCE}}(\boldsymbol{\ell}^{\mathrm{grp}}, \mathbf{y}^{\mathrm{grp}}; \boldsymbol{\pi}^{\mathrm{grp}}), \qquad \gamma = 0.5 .
$$

Training is full batch with Adam (learning rate 0.01, weight decay $5 \times 10^{-3}$, dropout
$p = 0.6$, gradient norm clipped at 1) for at most 200 epochs, keeping the epoch with the best
validation AUROC of `any_onset` (patience 20). Every patient passes messages
(**transductive**); only training labels enter $\mathcal{L}$; test labels are never loaded.

## 6. The self-supervised TemporalGraphAE (`models.TemporalGraphAE`, `models.graph_ae_loss`)

The same encoder (§5.1-5.2; dropout 0.5, weight decay $5 \times 10^{-4}$), with no demographics
and no labels, trained on two objectives.

**Link reconstruction.** In window $t$, let $\mathcal{E}_t$ be the undirected edges of
$\mathbf{W}_t$ and $\mathcal{E}^-_t$ as many node pairs drawn uniformly (fresh each epoch) among
nodes with an edge, redrawn while they are an edge. An inner-product decoder scores a pair by
$\sigma(\mathbf{z}_{i,t}^\top \mathbf{z}_{j,t})$:

$$
\mathcal{L}_{\mathrm{link}} = \frac{1}{T} \sum_{t} \frac{-1}{|\mathcal{E}_t| + |\mathcal{E}^-_t|}
\Big[\sum_{(i,j) \in \mathcal{E}_t} \log \sigma(\mathbf{z}_{i,t}^\top \mathbf{z}_{j,t}) + \sum_{(i,j) \in \mathcal{E}^-_t} \log\big(1 - \sigma(\mathbf{z}_{i,t}^\top \mathbf{z}_{j,t})\big)\Big].
$$

**Next window.** A linear decoder from the state after window $t$ predicts the codes active in
window $t + 1$, for $t = 1, \dots, T-1$ (2017→2018 to 2021→2022, all before the cutoff):

$$
\mathbf{q}_{i,t} = \mathbf{V} \mathbf{h}_{i,t} + \mathbf{c} \in \mathbb{R}^{C}, \qquad
\mathcal{L}_{\mathrm{next}} = \mathcal{L}_{\mathrm{BCE}}\big(\mathbf{q}_{i,t},\ \mathbf{B}_{t+1, i\cdot};\ \boldsymbol{\pi}^{\mathrm{code}}\big),
$$

with $\boldsymbol{\pi}^{\mathrm{code}}$ the per-code negatives-to-positives ratio on the training
split, and the target masked when patient $i$ is not yet born or already dead at the start of
window $t+1$. Only training-split patients' next-window terms enter the loss; validation
patients' pick the epoch; test patients enter only the link term (as nodes and edges). The loss
is

$$
\mathcal{L}_{\mathrm{AE}} = \mathcal{L}_{\mathrm{link}} + \mu\, \mathcal{L}_{\mathrm{next}}, \qquad \mu = 2
$$

($\mu \in \{0.5, 1, 2\}$ chosen by the lowest mean validation next-window loss over seeds 42-44).
Adam, learning rate 0.01, at most 300 epochs, early stopping on the validation next-window loss
(patience 20). The anti-collapse check is the share of variance of $\{\mathbf{h}_i\}$ on its first
principal component (`models.pc1_share`; floor $1/d$, collapse at 1).

## 7. Platt scaling (`evaluate.fit_platt`)

The `any_onset` logits $\ell$ are shifted upwards by $\pi$, so they are recalibrated on the
validation patients $\mathcal{V}$ (a held-out set never used for gradients):

$$
\hat p_i = \sigma(\alpha\, \ell_i + \beta), \qquad
(\alpha, \beta) = \arg\min_{\alpha, \beta} -\sum_{i \in \mathcal{V}} \big[y_i \log \sigma(\alpha \ell_i + \beta) + (1 - y_i) \log(1 - \sigma(\alpha \ell_i + \beta))\big],
$$

an unpenalized logistic regression of the label on the logit (BFGS from $(1, 0)$). The final
model's fit is $\alpha = 0.796$, $\beta = -1.272$. Platt scaling is monotone, so it changes
calibration (Brier, ECE) but not the AUROC.

## 8. Labels, masking and censoring (`labels.build_labels`)

Nothing on or after the horizon end $\tau + H$ (2026-01-01) is read. For onset group $g$ with
code set $\mathcal{C}_g$, the onset of patient $i$ is
$o_{ig} = \min\{d_e : e \text{ an event of } i \text{ with code in } \mathcal{C}_g\}$
($\infty$ if none). In order of precedence, the label is

| Condition | $y_{ig}$ |
| --- | --- |
| died before $\tau$ | masked |
| fewer than 2 distinct codes before $\tau$ (in the cohort only thanks to later codes) | masked |
| $o_{ig} < \tau$ (already has it: not at risk) | masked |
| lost to follow-up: alive through the horizon, last encounter before 2025-01-01 (rule a) | masked |
| $\tau \le o_{ig} < \tau + H$ | 1 |
| died in the horizon without the onset (rule b, censored) | masked |
| otherwise | 0 |

The primary target pools the groups: $y^{\mathrm{any}}_i = 1$ if some $y_{ig} = 1$; $0$ if some
$y_{ig}$ is observed and every observed one is 0; masked if all are masked. Masked labels enter
no loss and no metric. Because a group the patient already has is masked, patient $i$'s
**risk set** is $\{g : o_{ig} \ge \tau\}$: a multimorbid patient is at risk in fewer groups.
The labels are new onsets after the cutoff; the features use only windows before it, so no
label is a function of the features.

Patients are split 70/15/15 into training, validation and test, stratified by phenotype
cluster $\times$ $y^{\mathrm{any}}$ (masked as its own value), seed 42.

## 9. A new patient (`inference.insert_node`, `inference.nystrom_extension`)

The new patient $x$'s features $\mathbf{f}_{x,t}$ are built with the saved IDF and SVD (§2).
In each window $x$ is linked to its $k$ most cosine-similar reference patients
$\mathcal{N}_k(x)$, with weights $w_j = \operatorname{sim}(x, j)$.

**Graph insertion.** The reference graph is kept; only the degrees of $x$'s neighbours change,
$\tilde d'_j = \tilde d_j + w_j$, and $\tilde d_x = 1 + \sum_j w_j$. With
$\mathbf{R} = \operatorname{diag}\big(\sqrt{\tilde d / \tilde d'}\big)$ (different from $I$ only at
the neighbours),

$$
\hat{\mathbf{A}}' = \begin{bmatrix} \mathbf{R} \hat{\mathbf{A}} \mathbf{R} & \mathbf{a} \\ \mathbf{a}^\top & 1 / \tilde d_x \end{bmatrix},
\qquad a_j = \frac{w_j}{\sqrt{\tilde d'_j\, \tilde d_x}}\ \ (j \in \mathcal{N}_k(x)),\ 0 \text{ otherwise},
$$

which equals §4.2 applied to the augmented graph with only $x$'s own $k$ edges. The trained
models are applied with frozen weights to $\hat{\mathbf{A}}'$ and only $x$'s outputs are read.

**Nystrom extension.** Row $i$ of $\mathbf{L}\mathbf{u} = \lambda \mathbf{u}$ is
$u_i = \frac{1}{1 - \lambda} \sum_j \frac{W_{ij}}{\sqrt{D_{ii} D_{jj}}}\, u_j$. Applying it to the
new row gives $x$'s spectral coordinates without recomputing any eigenvector:

$$
u_{m,x} = \frac{1}{1 - \lambda_m} \sum_{j \in \mathcal{N}_k(x)} \frac{w_j}{\sqrt{d_x\, D_{jj}}}\, u_{m,j}, \qquad d_x = \sum_j w_j,
$$

with $D_{jj}$ the reference degrees. It is exact for a reference patient's own row and an
approximation for a new one (adding $x$ would perturb the eigenvectors slightly). In the
2022 spectral embedding, $x$ gets the phenotype of the nearest k-means centroid and its most
similar reference patients by cosine similarity, whose observed 2023-2025 onsets are reported.

## 10. Evaluation protocol

**Cross-validation (Phase 7, the main comparison).** The $n = 2{,}221$ patients with an observed
$y^{\mathrm{any}}$ are split into 5 folds stratified by phenotype $\times$ label (seed 42). In
each fold, 15% of the training folds (stratified the same way) form an inner validation set,
which picks the GCNs' epoch and fits each model's Platt scaling; models are fitted on the other
85%. Each GCN is trained with seeds 42, 43, 44 and its logits averaged. The held-out fold is
scored once, and the out-of-fold probabilities of all folds are pooled. Eight models: age and
sex only; logistic regression on the 2022 window or on all 6 windows (with age and sex); the
16-dimensional spectral embedding with logistic regression; a static GCN on 2022; the temporal
model with $\hat{\mathbf{A}}_t = I$ (no graph); the TemporalGCN on degree-preserving rewired
graphs (10 double edge swaps per edge); and the full TemporalGCN.

**Held-out test set.** The final model of Phase 6 (trained on the training split, epoch and
calibration on validation) is scored once on the 330 labelled test patients.

**Metrics.** AUROC (the probability that a random positive outranks a random negative);
AUPRC; Brier score $\frac{1}{n}\sum_i (\hat p_i - y_i)^2$; expected calibration error over $B = 10$
equal-count bins $\mathcal{B}_b$, $\mathrm{ECE} = \sum_b \frac{|\mathcal{B}_b|}{n} \big|\bar{\hat p}_b - \bar y_b\big|$.

**Paired percentile bootstrap.** For $r = 1, \dots, R$ ($R = 1{,}000$, seed 42), draw $n$
patients with replacement, $\mathcal{I}_r$, and compute every model's metric on the **same**
resample. A 95% interval is the 2.5th and 97.5th percentiles of $\{M(\mathcal{I}_r)\}_r$; the
difference between models A and B uses $\{M_A(\mathcal{I}_r) - M_B(\mathcal{I}_r)\}_r$, so the
shared patient sampling cancels. A difference is **shown** when its interval excludes 0. The
intervals reflect patient sampling, not retraining (seed) variation.

**Neighbour concordance (Phase 7b).** With $S_i$ the set of chronic conditions active for
patient $i$ in 2017-2022 and $\mathcal{N}^{\mathrm{emb}}_5(i)$ its 5 nearest patients by cosine
distance in an embedding,

$$
\mathrm{conc}(i) = \frac{1}{5} \sum_{j \in \mathcal{N}^{\mathrm{emb}}_5(i)} \frac{|S_i \cap S_j|}{|S_i \cup S_j|},
$$

averaged over the test patients with at least one chronic condition. Chance (5 random
patients) is 0.103.

**Next-year task (Phase 7b).** Patient $i$ is at risk for code $c$ in 2022 if alive at its start
and with no event of $c$ before it; the label is 1 if the first event of $c$ is in 2022. For
each of the 12 codes with at least 20 training onsets, a logistic regression on the 2021
embedding is fitted on training patients and scored on test patients. Micro AUROC pools every
at-risk (patient, code) pair; macro AUROC averages the per-code AUROCs. The reference is each
code's training base rate, which carries no patient information (micro AUROC 0.760).
