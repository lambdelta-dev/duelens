# duelens: choosing exemplar pairs for few-shot decision APIs

**A short paper for the idea.** The implementation lives in this repository; `README.md` is the practical manual.

---

## Abstract

A class of image decision APIs is conditioned on a small set of reference images: one or more labeled "good" examples and one or more labeled "bad" examples. The service then returns a score, class, or segmentation for a new image relative to those references. When exactly one good and one bad reference are allowed, the choice of pair is a high-leverage, purely combinatorial decision that is usually made by hand. We formalise the choice as maximising the separability of the labeled set under a reference-relative scoring function, and describe `duelens`, a Rust tool that searches the pair space, evaluates each pair with leave-two-out AUC and related statistics, and reports calibrated scores, uncertainty, and the winning pair. The default scorer is a dependency-free classical-CV descriptor; the search operates on a distance matrix and can be reused with stronger embeddings or an API-backed scorer. On a synthetic circles-versus-rectangles benchmark the tool evaluates all 324 pairs in milliseconds and finds a pair with AUC 1.00 where the median pair scores 0.66 and the worst scores 0.44, illustrating both the value of the search and the variance hidden in a small labeled set.

## 1. Problem

Let

* `G = {g_1, ..., g_P}` be images labeled *good*,
* `B = {b_1, ..., b_N}` be images labeled *bad*,
* `D` be a decision API that, given references `(g, b)` and an image `x`, returns a good-vs-bad score, probability, or segmentation.

We must choose

```text
(g*, b*) = argmax_{g ∈ G, b ∈ B}  Q(D; g, b, G ∪ B)
```

where `Q` measures how well the references make the API separate the labeled good images from the labeled bad ones. The API is expensive or unavailable during development, so `duelens` substitutes a differentiable-free **surrogate**: a visual descriptor and a reference-relative distance score. The search and evaluation machinery is scorer-agnostic; only the descriptor is a stand-in.

The key structural point is that the two choices are coupled. The best good exemplar depends on which bad exemplar it is contrasted with, and vice versa. Selecting a class medoid independently is a heuristic, not a solution. The pair space is `P · N`, small enough for exhaustive search in many practical curation sets and large enough to need a heuristic in others.

## 2. Surrogate scoring

### 2.1 Descriptor

Each image is center-cropped to a square, resized to 64 × 64, and mapped to a 503-dimensional vector `φ(x)`. Five blocks are L2-normalized separately, weighted, concatenated, and renormalized:

| block | dim | weight |
|---|---:|---:|
| 16 × 16 luma thumbnail | 256 | 1.0 |
| 4 × 4 × 4 RGB histogram (√ frequencies) | 64 | 0.9 |
| HOG-lite, 4 × 4 cells, 8 unsigned orientations | 128 | 1.2 |
| 4 × 4 spatial RGB grid | 48 | 0.9 |
| global statistics | 7 | 0.7 |

The gradient block uses Sobel responses with per-cell L1 normalization. The global statistics are mean and standard deviation of luma, mean saturation, Hasler–Süsstrunk colorfulness, edge density, gradient-orientation entropy, and center-surround luma contrast. The histogram is Hellinger-transformed (square root of frequencies) so that Euclidean distance approximates a distribution distance.

This descriptor captures layout, palette, and coarse shape. It is not semantic and is intentionally cheap. Let `d(x, y) = ||φ(x) − φ(y)||₂`.

### 2.2 Reference-relative score

For a candidate pair `(g, b)`, every image receives

```text
s_{g,b}(x) = d(x, b) − d(x, g).
```

A larger score means closer to the good reference than to the bad reference. Under a nearest-reference model, the implied good probability is monotone in `s`, so ranking quality and thresholded accuracy are the natural objectives.

### 2.3 Leave-two-out evaluation

If the references themselves are included, `s(g) = d(g,b) − 0` and `s(b) = 0 − d(b,g)` are trivially ordered, inflating every pair. We therefore evaluate on

```text
H(g, b) = (G ∪ B) \ {g, b}.
```

This is stricter than leave-one-out because both reference images are removed. With `P, N ≥ 2` at least one image of each class remains. The tool reports AUC including the exemplars separately, as a sanity check only.

### 2.4 Objectives

Let `{s_i}` be the held-out scores and `y_i ∈ {0,1}` their labels.

**AUC** (default) is the Mann–Whitney statistic

```text
AUC = P(s_good > s_bad) + ½ P(s_good = s_bad),
```

computed by average ranks in `O(n log n)`. It is threshold-free and directly measures the probability that a random good image outranks a random bad image.

**Balanced accuracy** sweeps every distinct cut `t` (`good` iff `s > t`) and keeps

```text
BA = max_t ½ (TPR(t) + TNR(t)).
```

The selected threshold is the midpoint between adjacent scores at the optimum. This mirrors the operating point a deployed API would need.

**Cohen's d** is the standardised mean gap

```text
d = (μ_good − μ_bad) / σ_pooled,
```

clamped to ±100 when a class has zero variance. It rewards margin rather than ordering alone.

The optional **blend** objective is `0.5·AUC + 0.3·BA + 0.2·sigmoid(d)`. All four objectives are exposed as `--objective`.

### 2.5 Calibration

For the winning pair we fit a lightly regularised logistic model on the held-out scores,

```text
P(good | s) = σ(a·s + b),   σ(z) = 1 / (1 + e^{−z}),
```

by Newton's method with an L2 penalty of `1e−2` on the standardised slope and intercept. The fit is reported but never used to rank pairs: it is a read-out for humans and for API configuration, not part of the selection objective. When separation is near-perfect the slope is steep and probabilities saturate; the report exposes `a` and `b` so this is visible.

## 3. Search

### 3.1 Exhaustive

If `|G|·|B| ≤ M` (default `M = 200,000`), evaluate every pair. Distances are precomputed once into an `n × n` matrix, so each pair costs only the `O(n log n)` sort needed for AUC. Evaluation is parallelised with Rayon. The result is the exact optimum under the surrogate.

### 3.2 Beam search

Above the threshold, the tool uses a seeded beam:

1. Seed with the good-class medoid × bad-class medoid and uniformly sampled unique pairs until `max(64, 8·width)` seeds.
2. Evaluate and keep the best `width` pairs.
3. Expand each beam member by replacing its good reference with every `g ∈ G` and its bad reference with every `b ∈ B`, deduplicating against all previously evaluated pairs.
4. Keep the best `width` of the union; stop after two rounds without improvement or at the round cap.

The medoid term guarantees a sensible starting point; the expansions are coordinate ascent in both coordinates. The procedure is deterministic for a given `--seed`. It is approximate and the report says so.

### 3.3 Complexity

With `n = |G| + |B|` and descriptor dimension `D`:

* distance matrix: `O(n²D)` time, `O(n²)` memory;
* exhaustive: `O(|G||B| · n log n)`;
* beam: `O(R · W · (|G|+|B|) · n log n)`, with `R` rounds and width `W`.

At `n = 400` and 40,000 pairs the exhaustive path remains interactive on a laptop.

## 4. Experiment

We generated a synthetic benchmark of 36 images (18 good, 18 bad), 96 × 96 pixels. Good images contain a filled circle; bad images contain a filled rectangle. Background gradient, palette, shape color, and placement are sampled independently of the label, so shape is the only reliable cue. Five images per class are crisp, centerd prototypes; the other thirteen are blurred, low-contrast, and color-cast.

Command (release build):

```sh
duelens demo --seed 7
```

Results:

| quantity | value |
|---|---|
| candidate pairs | 324 |
| search | exhaustive |
| wall time | 6 ms |
| AUC, worst pair | 0.439 |
| AUC, median pair | 0.661 |
| AUC, best pair | 1.000 |
| winner balanced accuracy | 1.000 |
| winner Cohen's d | 4.429 |
| bootstrap 95% AUC interval | [1.000, 1.000] |

The spread is the headline: with the same images, the same descriptor, and the same API contract, the reference choice moves AUC from below chance to perfect separation. The winning pair was not the most aesthetically "clean" pair; near-perfect separation held for several pairs, and the tie-break favoured the larger standardised margin. Inspecting the winner reveals a more interesting effect: both references share a green palette even though color is sampled independently of the label. The pair therefore approximately cancels color as a nuisance cue and leaves shape as the discriminative signal — a paired control discovered by the objective, not by any explicit color-matching term. A rough pair can also help because blur suppresses idiosyncratic background detail. Photographic quality is not the target; induced contrast is.

The bootstrap interval degenerates at AUC = 1, which is a property of percentile intervals at a boundary, not evidence of certainty. With 36 images, the selection of the best of 324 pairs also carries a winner's-curse bias; the interval is reported as a sampling-noise band for the chosen pair, not as a corrected estimate of its generalisation.

## 5. Discussion and limitations

The method is only as good as the surrogate. A classical descriptor can rank pairs sensibly on clean, object-centric data and fail on fine-grained or semantic distinctions that a real API would capture. The intended workflow is therefore a **shortlist-and-validate** loop: run `duelens`, take the top pairs (plus the exported pair), and measure them through the real decision API. The search code does not care where the distance matrix came from; replacing the descriptor with CLIP/DINO embeddings or API-derived scores is the obvious next step.

Other limitations are explicit. The tool selects exactly one good and one bad reference, matching a narrow API contract. It treats images independently and ignores segmentation masks. It does not handle class imbalance beyond using balanced metrics. Beam search trades optimality for scale. Small labeled sets make AUC estimates noisy, and the confidence interval does not account for selection.

Two design choices are worth keeping when extending the tool. First, always evaluate references out-of-sample; the leave-two-out rule is what makes the numbers meaningful. Second, report the **distribution** of pair quality, not just the winner. The demo's worst-median-best spread is the practical argument for the tool, and it tells the user whether curation is worth the effort.

Future work includes embedding inputs, direct API-in-the-loop scoring with caching, k-shot selection, joint threshold optimization, and selection-bias-corrected evaluation via nested cross-validation.

## 6. Reproduction

```sh
cargo run --release -- demo --seed 7
cargo run --release -- find .duelens-demo --objective blend --top 10
cargo test
```

Artifacts: `.duelens-demo/report.md`, `.duelens-demo/picked/pair.json`, and the full terminal trace.

## References

* Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *Annals of Mathematical Statistics*, 18(1), 50–60.
* Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum.
* Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In *Advances in Large Margin Classifiers*. MIT Press.
* Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. *CVPR*.
* Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
