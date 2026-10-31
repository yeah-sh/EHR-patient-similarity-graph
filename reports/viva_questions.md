# Viva questions

Twenty questions an examiner is likely to ask, with short answers from our actual results.
Brackets are 95% percentile bootstrap intervals over patients. Sources: `findings.md`,
`results.md`, `embedding_results.md`, `graphs_summary.md`, `labels_summary.md`, `methods.md`.

## The design

### 1. Where is the time-series?

In four places. (1) Each patient is a sequence of six yearly feature vectors, 2017-2022, built
from the conditions active in each year; the features themselves carry history through the
time-decayed sum $\sum_{s \le t} e^{-0.1 (t - s)} X_s$ and the transitions up to that year.
(2) There is one similarity graph per year, and the graphs drift: neighbourhood overlap rises
from 0.405 (2017 to 2018) to 0.604 (2021 to 2022). (3) The shared GCN gives a 32-dimensional
embedding per patient per year (`embeddings_per_window.npy`, N x 6 x 32), and a GRU reads them in
order; its state after each year is the trajectory, and its final state the trajectory
embedding. (4) The self-supervised GraphAE is trained to predict year t + 1's conditions from
the state after year t. Honestly, though, time did not help forecasting: removing it (a static
GCN on 2022 only) changes the cross-validated AUROC by -0.023 (-0.057 to +0.010), not shown.

### 2. What is linear-algebraic about it?

Almost every step before the neural network is a matrix operation on sparse matrices: TF-IDF
weighting ($B \operatorname{diag}(\mathrm{idf})$), the time-decay recursion, per-patient Markov
transition matrices compressed by truncated SVD, co-occurrence counts $Z^\top Z$ and the lift,
cosine similarity of row-normalized vectors, the normalized adjacency
$\hat A = \tilde D^{-1/2}(W + I)\tilde D^{-1/2}$ and Laplacian $L = I - D^{-1/2} W D^{-1/2}$, the
Lanczos eigendecomposition of $L$, the eigengap, spectral clustering, and the Nystrom extension
for new patients. The GCN layer itself is $\hat A H \Theta$: a learned linear map followed by
graph averaging. No dense $N \times N$ matrix is ever built.

### 3. Why aren't your labels circular?

Because the labels are about a different time from the features. Features use only the windows
2017-2022, which close before the cutoff 2023-01-01 (transitions are cut at the cutoff too);
the label is a *new* onset in 2023-2025 of one of 9 chronic groups. A group the patient already
had before the cutoff is masked, not counted as positive, so the model cannot score by
recognising a condition that is already in its input. The phenotype clusters are used only to
stratify the split, never as a target. Nothing on or after 2026-01-01 is read.

### 4. Is there any leakage?

Not of labels. Test labels are set to NaN when training data are loaded, validation labels only
choose the epoch and the calibration, and every feature predates the cutoff. Two things are
shared on purpose and stated as limitations: training is **transductive**, so test patients
are graph nodes with their pre-cutoff features during training (never their labels); and the
unsupervised transforms (IDF, transition SVD) and the GraphAE's link term use every patient's
pre-cutoff data, test patients included. The second may favour graph-based embeddings in the
test-patient concordance. One mild optimism is documented: the GCN hyperparameters were tuned
on the validation split, whose patients are also in the cross-validation folds.

### 5. Why do the windows start in 2017?

Because Synthea exports a complete record only for the 10 years before its run date
(2026-10-30). Clinical events dated per year are about 400 before 2017 and about 2,500 from
2017 on (`figures/final/03_active_patients_per_year.png`), and 71% of clinical events before
2016-10-30 have no STOP date against 14% after: earlier years keep mostly conditions that were
still open. Earlier windows would miss most acute conditions. Conditions that began earlier and
are still active are carried into 2017 by their active span.

### 6. Why cosine k-NN with k = 10, and why symmetrize with max?

The feature rows are non-negative and L2-normalized, so cosine similarity compares the mix of
conditions regardless of how many a patient has, and lies in [0, 1]. A k-NN graph keeps it
sparse (about 19,000 edges for 2,581 patients, mean degree 14.7 in 2022) and local, and
symmetrizing with max keeps an edge if either patient chose the other, so no patient with
features has fewer than 10 neighbours. By 2022 the graph is one connected component of all
2,581 patients. k = 10 was set a priori and not tuned; a sensitivity analysis over k is not in
this project.

## The graph and the embeddings

### 7. How do you know the similarity graph carries clinical information?

By a negative control. The same self-supervised GraphAE trained on degree-preserving rewired
graphs (every patient keeps their number of neighbours, partners random) drops from 0.358
(0.332-0.381) to 0.123 (0.111-0.134) neighbour concordance, near chance at 0.103: a difference
of +0.235 (+0.213 to +0.256). With no graph at all it reaches 0.318. So it is who is linked to
whom that carries the structure, not the degree pattern.

### 8. What does the Laplacian spectrum tell you? Are the 7 phenotypes real clusters?

The largest gap among the 20 smallest eigenvalues of the 2022 Laplacian follows the 7th, so
spectral clustering uses 7 phenotypes, and their profiles are coherent: chronic pain in 98% of
cluster 4, obstructive sleep apnoea in 98% of cluster 7 (40.7 times its overall rate),
diabetic and cardiac complications in cluster 2. But the spectrum rises smoothly (0.028 at
$\lambda_2$ to 0.126 at $\lambda_{20}$), there is no sharp gap, and the silhouette of the
phenotypes in the GraphAE space is -0.012 (shuffled labels -0.181). They are useful partitions
of a continuum, not separate communities.

### 9. Why a GCN rather than plain k-NN?

A k-NN method uses the neighbours as they are: average their labels, or list them. A GCN learns
which features to propagate and how to combine a patient's own features with their
neighbourhood's, it gives an embedding per patient per year that a GRU can read in order, and it
can be trained without labels (the GraphAE). Honestly, on this data the extra machinery did not
pay off for prediction: the temporal model without a graph scores 0.583 against 0.579 with it.
For similarity search we actually use a linear method, the spectral embedding of the k-NN graph
(question 14), and the app lists the k nearest reference patients directly.

### 10. Why is the GCN no better than logistic regression?

Because nothing is much better than anything here. In 5-fold cross-validation on 2,221 patients
(414 positives) all 8 models lie between 0.553 and 0.585 pooled AUROC: logistic regression on
the 2022 window with age and sex 0.585 (0.556-0.614), the full TemporalGCN 0.579 (0.549-0.608),
age and sex alone 0.554 (0.525-0.582). No difference with the TemporalGCN is shown, including
removing the graph (-0.003) or time (+0.023). When the signal is weak, a model with more
parameters cannot find structure that is not there, and it pays in variance. Our reading is a
limit of the data: Synthea's rule-based modules seem to leave earlier conditions little
information about later onsets beyond age and sex (age and sex alone reach 0.88 for ischaemic
heart disease). That reading is not tested here.

### 11. Why did validation say 0.68 but test 0.56?

Three reasons. (1) Selection: the validation split chose the hyperparameters (best of 48
settings, by mean validation AUROC 0.677 over three seeds) and the epoch, so its 0.678 is
optimistic. (2) Noise: the splits are small (63 and 61 positives); the test interval is
0.483-0.632. (3) Which patients: out of fold, models trained on other patients score 0.633 on
the validation patients and 0.640 on the test patients, so the single-split test 0.559 is at
the low end of a wide spread. The number to quote is the pooled cross-validated AUROC, 0.579
(0.549-0.608), which uses every labelled patient.

### 12. Why did the supervised embedding collapse?

Its only training signal is one binary target. The head needs just one direction in the
32-dimensional state to separate positives from negatives, and nothing rewards the other
directions: 99.8% of the variance of the supervised TemporalGCN's final states lies on the first
principal component. A one-dimensional embedding cannot preserve who is similar to whom, so its
neighbour concordance is 0.143 (0.130-0.156), barely above chance. Its linear-probe AUROC (0.691)
looks good only because it was trained on those labels; its clean cross-validated AUROC is
0.579.

### 13. Why self-supervised?

Because the labels are few (414 positives), near chance to predict, and masked for many
patients, while the graph and the yearly records exist for all 2,581 patients. The GraphAE
learns from structure everyone has: reconstructing each year's similarity edges and predicting
next year's conditions. It does not collapse (64.7% of variance on the first component, against
99.8%) and keeps far more clinical similarity (0.358 against 0.143). It is still beaten by the
linear methods, and adding the next-window objective lowered concordance (-0.046, -0.062 to
-0.030) compared with the link term alone; its value is the per-year state, the trajectory.

### 14. Why spectral over the neural embedding for similarity search?

It preserves clinical similarity best: 0.431 (0.408-0.455) concordance against 0.358 for the
TemporalGraphAE, a shown difference of -0.074 (-0.095 to -0.055) for the GraphAE. It is also
deterministic (no training, no seed variance; the GraphAE's next-year AUROC moves between 0.741
and 0.755 across seeds), interpretable through its eigenvalues and eigengap, and it extends to a
new patient in closed form (Nystrom). Two caveats: its interval overlaps PCA of every window
(0.414; that difference was not tested), and none of the embeddings reaches the 236-dimensional
input features themselves (0.503). So the app searches neighbours in the spectral embedding and
draws trajectories with the GraphAE, each doing the job it does best.

### 15. Why does a healthy patient show a higher neighbour onset rate than a multimorbid one?

The risk set. `any_onset` is a *new* onset in any of 9 groups, and a group the patient already
has is masked. A multimorbid patient, and their similar patients, already have many groups and
are at risk in few; a healthy patient's neighbours are at risk in all 9. In `predict_demo.md` a
healthy young adult (at risk for 9 of 9 groups) has 3/8 neighbours with an onset, against 3/10
for a multimorbid elderly patient (at risk for 5). That is more groups in which an onset can
occur, not more disease: `any_onset` measures new onsets, not disease burden. Each prediction
reports its risk set ("at risk for 5 of 9 groups") for this reason.

### 16. How do you place a new patient? What is the Nystrom extension?

Nothing is refitted. The patient's features use the saved IDF and SVD. In each year's graph
they become node N + 1 linked to their 10 most similar reference patients, and only the
affected rows of $\hat A$ are re-normalized; the trained models run with frozen weights. For the
spectral embedding, row i of $L u = \lambda u$ gives
$u_i = \frac{1}{1 - \lambda} \sum_j \frac{W_{ij}}{\sqrt{d_i d_j}} u_j$; applied to the new row it
places the patient as a degree-weighted average of their neighbours' eigenvector entries,
divided by $1 - \lambda$. It is exact for a reference patient's own row and an approximation for
a new one, because adding the patient would change the eigenvectors slightly. The test suite
re-enters 40 test patients as new patients (their own node excluded) and compares the result
with the transductive output: the `any_onset` probability differs by a median of 0.013 (max
0.050, correlation 0.991), and the GraphAE's per-year states have a median cosine similarity
of 0.998 (min 0.970).

### 17. Why Platt scaling, and why is the calibrated Brier score no better than the base rate?

Training with a positive-class weight (negatives / positives, about 4.4) pushes the logits up:
uncalibrated, the mean predicted probability on the test set is 0.422 against an observed rate
of 0.185. Platt scaling, $\sigma(0.796\, z - 1.272)$ fitted on validation, brings it to 0.192
and the ECE from 0.238 to 0.075. It is monotone, so the AUROC is unchanged (0.559). The Brier
score, 0.155 (0.129-0.180), cannot beat always predicting the base rate (0.151) by much when
the model barely ranks patients: calibration fixes the scale, not the discrimination.

## Reliability and the data

### 18. How sure are you of these numbers?

Each interval is a 95% percentile bootstrap over patients (1,000 resamples, seed 42), and
models are compared on the same resamples, so differences are paired; we call a difference
shown only when its interval excludes 0. The intervals cover patient sampling, not retraining:
GCNs in cross-validation are averaged over three seeds, and the GraphAE's seed sensitivity is
reported separately. The counts are small: 61 test positives, 31-109 positives per onset group
across all folds and 4-17 per group on the test set, 122 next-year onsets over 12 codes. We
make many comparisons without a multiplicity correction, so group-level results are indicative
only. The whole pipeline reruns from scratch with one command (`scripts/run_all.py`) and
reproduces the reported numbers.

### 19. Why Synthea, and what are its limits?

Why: it is open (no data-access approval), it uses real code systems (SNOMED-CT), records are
complete for every patient, and it let us build and test the entire pipeline reproducibly.
Limits: disease progression comes from fixed, rule-based modules, so dependencies between
conditions in different modules are weak or absent; the export keeps only a 10-year history; the
population is simulated, so prevalences and care patterns are not those of a real health system;
and we used only conditions (no medications, observations or notes). Our results describe how
the methods behave on such data. The near-chance forecasting is most likely a property of
Synthea, not evidence that the methods fail on real records.

### 20. What would you do with real data?

Run the same pipeline on MIMIC-IV, or better a primary-care EHR with long follow-up, with ICD-10
codes grouped into phecodes, and validate on an external cohort. Then test the question Synthea
could not answer: whether real temporal dependencies between conditions let the graph and the
temporal model beat age, sex and a logistic regression. Methodologically: train inductively
(patients held out of the graph) so that new patients are scored as training patients were;
model time to onset per group with the risk set explicit instead of a pooled binary target;
try attention over the yearly states instead of the GRU; and check calibration and subgroup
performance before any clinical use.
