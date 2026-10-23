# duelens

**duel + lens** — pick the good/bad reference pair that draws the sharpest decision boundary.

A few-shot decision API (JEV-style "show it one good image and one bad image") is only as good as the two images you hand it. `duelens` takes a labeled image set, scores every feasible `(good example, bad example)` pair, and returns the pair that best separates the two classes on the images it did *not* see as exemplars.

It is the System‑1 gut check for your reference set: one good anchor, one bad anchor, chosen by search instead of vibes.

```
18 good / 18 bad  ·  exhaustive · 324 pairs evaluated · 6 ms
pair-choice effect: AUC 0.439 (worst) · 0.661 (median) · 1.000 (best)

WINNER
  good   .duelens-demo/good/good_011_rough.png
  bad    .duelens-demo/bad/bad_016_rough.png
  AUC    1.000  [1.000, 1.000] 95% CI
  bal. acc 1.000   Cohen's d 4.429   threshold -0.0094
```

That is the real output of `duelens demo`: the best pair is perfect on this dataset, the median pair is mediocre, and the worst pair is *worse than a coin flip*. Choosing references randomly is leaving performance on the table.

---

## Quick start

```sh
cd duelens
CARGO_HOME="$PWD/.cargo-home" cargo run --release -- demo
# or, since a release binary is already built:
./target/release/duelens demo
```

In a normal environment drop the `CARGO_HOME` prefix; it is only needed here because the sandbox keeps the global Cargo cache read-only.

`demo` writes a synthetic circles-vs-rectangles dataset to `.duelens-demo/`, runs the search, and writes:

* `.duelens-demo/report.md` — full report,
* `.duelens-demo/picked/good.png` and `bad.png` — the winning pair,
* `.duelens-demo/picked/pair.json` — the same decision as machine-readable data.

On your own images:

```sh
CARGO_HOME="$PWD/.cargo-home" cargo run --release -- find path/to/dataset
```

The three accepted layouts:

```text
dataset/                  dataset/                       labels.csv
├── good/                 ├── good_001.png              path,label
│   ├── 001.jpg           ├── good_002.png              imgs/1.jpg,good
│   └── 002.jpg           ├── bad_001.png               imgs/2.jpg,bad
└── bad/                  └── bad_002.png
    ├── 101.jpg
    └── 102.jpg
```

```sh
# subdirectories
duelens find dataset

# flat prefixes: <label>_ <label>- <label>. <label><space>
duelens find dataset --good-label keep --bad-label reject

# explicit manifest (paths relative to the CSV)
duelens find dataset --manifest labels.csv
```

Only `png`, `jpg`/`jpeg`, `webp` and `bmp` are decoded. Images that fail to decode are reported and skipped.

---

## What it does

For a candidate pair `(g, b)`:

1. Compute a descriptor `φ(x)` for every image.
2. Score every held-out image by **contrast against the two references**:

   ```text
   s(x) = d(x, b) − d(x, g)          d = Euclidean distance
   ```

   A high score means "closer to the good exemplar than to the bad one".
3. Measure how well those scores separate the labeled classes:
   * **AUC** — `P(s_good > s_bad)`; threshold-free, the default objective.
   * **balanced accuracy** at the best threshold — what a single cutoff would achieve.
   * **Cohen's d** — effect size / margin between the two score clouds.
   * optionally a weighted blend.
4. Repeat for every pair and return the best.

The two exemplars are always **excluded from their own evaluation** (leave-two-out). Otherwise an exemplar trivially votes for itself and every pair looks perfect.

### Why this is a search problem

The number of pairs is `|G| × |B|`. Each pair's score depends non-linearly on both choices, so you cannot pick the best good image and the best bad image independently. `duelens` handles the permutation:

| candidate pairs | strategy |
|---:|---|
| ≤ `--max-exhaustive` (default 200,000) | **exhaustive** — provably best under the surrogate |
| more | **beam search** — seeded by class medoids + random samples, then repeated one-side swaps around the best beam (deterministic per `--seed`) |

For 200 images per class (40,000 pairs, 160,000 image-pair evaluations) exhaustive search is still fast because the full distance matrix is computed once and reused.

---

## The descriptor

The default scorer is a fast classical-CV descriptor, not a neural embedding. It exists so the tool runs anywhere with no model download, and it is honest about being a **stand-in** for your decision API.

| block | dimensions | what it captures | weight |
|---|---:|---|---:|
| luma thumbnail (16×16) | 256 | layout / coarse shape | 1.0 |
| RGB histogram (4×4×4, Hellinger) | 64 | palette | 0.9 |
| HOG-lite (4×4 cells × 8 orientations) | 128 | edges / structure | 1.2 |
| spatial color (4×4 cells × RGB) | 48 | where colors sit | 0.9 |
| global stats | 7 | brightness, contrast, saturation, colorfulness, edge density, orientation entropy, center-surround contrast | 0.7 |

Every image is center-cropped to a square, resized to 64×64, each block is L2-normalized and weighted, and the concatenated vector is L2-normalized once more. The descriptor is deliberately resolution-independent (a test covers this). Version: `classical-v1`.

The search only sees a distance matrix. Swapping in CLIP, DINO, or your own API-backed scorer is a contained change (see [Roadmap](#roadmap)).

---

## CLI

```
duelens find [PATH] [OPTIONS]
duelens demo [OPTIONS]
```

| flag | default | meaning |
|---|---|---|
| `--good-label` | `good` | directory name or filename prefix for the good class |
| `--bad-label` | `bad` | directory name or filename prefix for the bad class |
| `--manifest FILE` | — | CSV with `path,label` columns; overrides directory discovery |
| `--recursive` | off | recurse below the label directories |
| `--objective` | `auc` | `auc`, `balanced`, `effect`, `blend` |
| `--top N` | `10` | ranked pairs to report |
| `--seed N` | `42` | seed for beam sampling and bootstrap |
| `--max-exhaustive N` | `200000` | switch to beam search above this many pairs |
| `--beam-width N` | `16` | beam width for large spaces |
| `--no-bootstrap` | off | skip the 400-resample AUC confidence interval |
| `--json FILE` | — | write full machine-readable result |
| `--report FILE` | — | write a Markdown report |
| `--export DIR` | — | copy winner as `good.*`, `bad.*`, plus `pair.json` |
| `--quiet` | off | suppress stderr progress |

`--objective`:

* `auc` — threshold-free ranking quality; best default for "high-probability segmentation".
* `balanced` — maximise accuracy at the best single cutoff.
* `effect` — maximise Cohen's d (separation margin).
* `blend` — `0.5·AUC + 0.3·balanced + 0.2·sigmoid(d)`.

Demo options: `--out DIR`, `--per-class N`, `--clean-per-class N`, `--size PX`, `--seed N`, `--force`.

### Outputs

**Terminal** — winner, top pairs, AUC spread over all candidate pairs, a two-row score histogram (`▓` good, `░` bad), threshold and caveats.

**JSON** — everything:

```json
{
  "tool": "duelens",
  "version": "0.1.0",
  "config": { "objective": "auc", "seed": 7, "leave_two_out": true, "...": "..." },
  "dataset": { "n_good": 18, "n_bad": 18, "skipped": [] },
  "search": { "mode": "exhaustive", "pairs_possible": 324, "pairs_evaluated": 324,
              "auc_worst": 0.439, "auc_median": 0.661, "auc_best": 1.0 },
  "winner": {
    "good": ".duelens-demo/good/good_011_rough.png",
    "bad": ".duelens-demo/bad/bad_016_rough.png",
    "auc": 1.0, "auc_ci95": [1.0, 1.0], "balanced_accuracy": 1.0,
    "cohens_d": 4.429, "threshold": -0.0094, "auc_with_exemplars": 1.0
  },
  "top": [ "..." ],
  "winner_scores": [ { "path": "...", "label": "good", "held_out": true,
                       "d_good": 0.62, "d_bad": 0.80, "score": 0.18,
                       "p_good": 0.97 } ],
  "calibration": { "slope": 181.37, "intercept": 4.23,
                   "threshold": -0.0094, "threshold_probability": 0.463 },
  "notes": [ "..." ]
}
```

**Markdown** — the same result as a report with tables, the score distribution, calibration, and every held-out image's score.

**Export** — the two winning image files plus `pair.json`, ready to upload to the decision API.

---

## How the numbers are computed

### AUC

With `P` held-out good scores and `N` held-out bad scores, AUC is the Mann–Whitney statistic

```text
AUC = (Σ rank(s_good) − P(P+1)/2) / (P·N)
```

Ties get average ranks, so tied good/bad scores contribute ½. Perfect ordering = 1.0, reversed = 0.0, chance = 0.5.

### Balanced accuracy

For every distinct score cutoff `t` (predict good iff `s > t`), compute

```text
BA(t) = ½ · (TPR(t) + TNR(t))
```

and keep the best. The reported threshold is the midpoint between the two adjacent scores at that cut.

### Effect size

```text
d = (mean(s_good) − mean(s_bad)) / pooled_sd
```

### Calibration (report only, not used for ranking)

A lightly regularised logistic curve is fitted to the held-out scores:

```text
P(good | s) = sigmoid(a·s + b)
```

This is what a probability readout would look like for the chosen pair. Near-perfect separation makes the slope steep and the absolute probabilities optimistic; the report says so implicitly by showing the fit parameters.

### Bootstrap band

The winner gets a 95% percentile interval from 400 bootstrap resamples of its held-out scores. This measures sampling noise in the image set for a *fixed* pair. It does **not** correct for the fact that the pair was selected as the best of many (winner's curse); treat it as a sanity band.

### Complexity

* Distance matrix: `O(n² · D)` time, `O(n²)` memory.
* Exhaustive: `O(|G|·|B| · n log n)` time for sorting each pair's scores.
* Beam: `O(rounds · width · (|G|+|B|) · n log n)` time; approximate.
* Descriptor: `O(n · 64²)` scalar work after decode.

---

## Reading the demo

The synthetic set has circles (good) and rectangles (bad). Color, background gradient, and shape placement are randomized independently of the label, so shape is the only reliable cue. Some images are crisp "clean" prototypes; the rest are blurred, low-contrast, color-cast "rough" shots.

The search often picks rough images — and that is not a bug. A pair is judged by the **separation it induces**, not by how pretty the references are. On the default seed the winning pair is a rough green circle and a rough green rectangle: both references share a palette even though color is label-independent. Matching color cancels it as a nuisance cue and leaves shape as the separating signal. A rough image can also be a better contrast anchor because blur suppresses background detail. The search discovered a paired control; it was never told to match color.

What matters is the AUC spread: on the default demo, the worst pair is *below chance* and the median is 0.66 while the best is 1.00. That is the gap the search closes.

---

## Limitations

* **Surrogate, not your API.** The classical descriptor approximates visual similarity. A pair that wins here should be validated against the real decision API before production. The report says this too.
* **One pair only.** The API contract assumed here is exactly one good and one bad reference. k-shot selection is future work.
* **Cohen's d needs two held-out images per class.** With exactly two images per class, one of each is held out and `d` is reported as `n/a`; AUC and balanced accuracy still work.
* **Small-sample noise.** With tens of images, AUC estimates are noisy. The bootstrap CI quantifies some of it; selection bias remains.
* **Descriptor blind spots.** No semantics, no text, no fine-grained texture, no EXIF orientation handling. A photo with a rotated subject may be treated as a different scene.
* **Class balance.** AUC/balanced accuracy are insensitive to class ratio, but the candidate pool is not weighted; with very unbalanced sets the majority class dominates the number of possible pairs.
* **Beam search is approximate.** Exhaustive mode is exact; beam mode is a deterministic heuristic. The report labels the mode.

---

## Roadmap

* `--embeddings FILE` — plug in CLIP/DINO vectors from an existing pipeline.
* Real decision-API scorer — evaluate top candidates through the API itself, with caching and early stopping.
* k-shot / multi-exemplar selection, and joint optimization of the reference **and** the API's threshold.
* Grouped cross-validation and a selection-bias-corrected winner score.
* HTML contact sheet of the shortlist for quick human review.

---

## Development

```sh
CARGO_HOME="$PWD/.cargo-home" cargo test
CARGO_HOME="$PWD/.cargo-home" cargo clippy --all-targets
CARGO_HOME="$PWD/.cargo-home" cargo fmt --all
```

Layout:

```
src/
  main.rs      CLI
  dataset.rs   discovery: subdirs, prefixes, manifest
  features.rs  classical-v1 descriptor
  metrics.rs   AUC, balanced accuracy, Cohen's d, Platt, bootstrap
  search.rs    distance matrix, exhaustive + beam, leave-two-out
  report.rs    terminal / Markdown / JSON / export
  synth.rs     circles-vs-rectangles demo generator
tests/e2e.rs   full pipeline + CLI + demo
PAPER.md       the idea on paper
```

Tests cover metric edge cases (ties, perfect/reversed separation), descriptor invariance and finiteness, all three discovery layouts, exhaustive and beam search, the synthetic pipeline, JSON/report/export output, and the CLI.

---

## The name

**duelens** = **duel** (good vs bad exemplar) + **lens** (images). It also nods to System 1: the decision API's fast, intuitive call is only as good as the two references it sees first. The bad exemplar is the "foil" in the duel; the search finds the sharpest one.
