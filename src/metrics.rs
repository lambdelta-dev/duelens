//! Separation metrics and calibration for a scored set of good/bad images.
//!
//! For an exemplar pair `(g, b)` every held-out image `x` gets a score
//!
//! ```text
//! s(x) = d(x, b) - d(x, g)
//! ```
//!
//! so a larger score means "closer to the good exemplar than to the bad one".
//! The pair is then judged by how well `s` orders the labelled good images above
//! the labelled bad images.

use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// Area under the ROC curve, i.e. `P(s_good > s_bad)`, with ties counted 1/2.
///
/// Returns `None` unless both classes are present.
pub fn roc_auc(scores: &[f64], labels: &[bool]) -> Option<f64> {
    if scores.len() != labels.len() {
        return None;
    }
    let n_pos = labels.iter().filter(|&&y| y).count();
    let n_neg = labels.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return None;
    }

    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));

    // Average ranks for ties, 1-based.
    let mut ranks = vec![0f64; scores.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i + 1;
        while j < order.len() && scores[order[j]] == scores[order[i]] {
            j += 1;
        }
        let avg = ((i + 1) as f64 + j as f64) / 2.0;
        for &idx in &order[i..j] {
            ranks[idx] = avg;
        }
        i = j;
    }

    let rank_sum: f64 = labels
        .iter()
        .zip(&ranks)
        .filter(|(y, _)| **y)
        .map(|(_, &r)| r)
        .sum();
    let auc = (rank_sum - (n_pos * (n_pos + 1)) as f64 / 2.0) / (n_pos * n_neg) as f64;
    Some(auc.clamp(0.0, 1.0))
}

/// Balanced accuracy at the best threshold.
///
/// The returned threshold `t` classifies `x` as good when `s(x) > t`.
pub fn best_balanced_accuracy(scores: &[f64], labels: &[bool]) -> Option<(f64, f64)> {
    if scores.len() != labels.len() {
        return None;
    }
    let n_pos = labels.iter().filter(|&&y| y).count();
    let n_neg = labels.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return None;
    }

    let mut data: Vec<(f64, bool)> = scores.iter().copied().zip(labels.iter().copied()).collect();
    data.sort_by(|a, b| a.0.total_cmp(&b.0).then(b.1.cmp(&a.1)));

    let mut best_ba = 0.5;
    let mut best_t = data[0].0 - 1.0;
    let mut pos_below = 0usize;
    let mut neg_below = 0usize;
    let mut i = 0;
    while i < data.len() {
        let s = data[i].0;
        while i < data.len() && data[i].0 == s {
            if data[i].1 {
                pos_below += 1;
            } else {
                neg_below += 1;
            }
            i += 1;
        }
        // Cut just above `s`: everything at or below is predicted bad.
        let tpr = (n_pos - pos_below) as f64 / n_pos as f64;
        let tnr = neg_below as f64 / n_neg as f64;
        let ba = 0.5 * (tpr + tnr);
        if ba > best_ba + 1e-12 {
            best_ba = ba;
            best_t = if i < data.len() {
                0.5 * (s + data[i].0)
            } else {
                s + 1.0
            };
        }
    }
    Some((best_ba, best_t))
}

/// Cohen's d effect size for good-minus-bad scores.
///
/// Returns `None` when either class has fewer than two held-out images (the
/// pooled variance is undefined). Exactly separated scores with zero
/// within-class variance are clamped to +/-100 so the value stays
/// JSON-representable.
pub fn cohens_d(scores: &[f64], labels: &[bool]) -> Option<f64> {
    if scores.len() != labels.len() {
        return None;
    }
    let good: Vec<f64> = scores
        .iter()
        .zip(labels)
        .filter(|(_, y)| **y)
        .map(|(&s, _)| s)
        .collect();
    let bad: Vec<f64> = scores
        .iter()
        .zip(labels)
        .filter(|(_, y)| !**y)
        .map(|(&s, _)| s)
        .collect();
    if good.len() < 2 || bad.len() < 2 {
        return None;
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let var =
        |v: &[f64], m: f64| v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
    let mg = mean(&good);
    let mb = mean(&bad);
    let pooled = ((var(&good, mg) + var(&bad, mb)) / 2.0).sqrt();
    let diff = mg - mb;
    if pooled < 1e-12 {
        Some(if diff > 0.0 {
            100.0
        } else if diff < 0.0 {
            -100.0
        } else {
            0.0
        })
    } else {
        Some((diff / pooled).clamp(-100.0, 100.0))
    }
}

/// One-dimensional logistic calibration of `s(x)` into `P(good | s)`.
#[derive(Debug, Clone, Copy)]
pub struct Platt {
    pub slope: f64,
    pub intercept: f64,
}

impl Platt {
    pub fn probability(&self, score: f64) -> f64 {
        sigmoid(self.slope * score + self.intercept)
    }
}

/// Fit a lightly regularised logistic curve. Returns `None` for degenerate
/// inputs (fewer than four points, or only one class).
pub fn fit_platt(scores: &[f64], labels: &[bool]) -> Option<Platt> {
    if scores.len() < 4 || labels.iter().all(|&y| y) || labels.iter().all(|&y| !y) {
        return None;
    }
    let n = scores.len() as f64;
    let mean = scores.iter().sum::<f64>() / n;
    let var = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n;
    let scale = var.sqrt().max(1e-6);
    let z: Vec<f64> = scores.iter().map(|s| (s - mean) / scale).collect();

    let lambda = 1e-2;
    let mut a = 0.0f64;
    let mut b = 0.0f64;
    for _ in 0..60 {
        let mut g0 = lambda * a;
        let mut g1 = lambda * b;
        let mut h00 = lambda;
        let mut h01 = 0.0;
        let mut h11 = lambda;
        for (&zi, &y) in z.iter().zip(labels) {
            let p = sigmoid(a * zi + b);
            let w = (p * (1.0 - p)).max(1e-9);
            let r = p - if y { 1.0 } else { 0.0 };
            g0 += r * zi;
            g1 += r;
            h00 += w * zi * zi;
            h01 += w * zi;
            h11 += w;
        }
        let det = h00 * h11 - h01 * h01;
        if det.abs() < 1e-12 {
            break;
        }
        let da = (h11 * g0 - h01 * g1) / det;
        let db = (h00 * g1 - h01 * g0) / det;
        a -= da;
        b -= db;
        if da.abs().max(db.abs()) < 1e-10 {
            break;
        }
    }
    Some(Platt {
        slope: a / scale,
        intercept: b - a * mean / scale,
    })
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Percentile bootstrap confidence interval for an AUC.
///
/// This quantifies sampling noise in the held-out images for a *fixed* pair; it
/// does not correct for the fact that the pair itself was selected as the best
/// of many. Use it as a sanity band, not a significance test.
pub fn bootstrap_auc_ci(
    scores: &[f64],
    labels: &[bool],
    seed: u64,
    resamples: usize,
) -> Option<(f64, f64)> {
    if scores.len() != labels.len() || scores.len() < 4 {
        return None;
    }
    let mut rng = SmallRng::seed_from_u64(seed);
    let n = scores.len();
    let mut aucs = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        let mut ss = Vec::with_capacity(n);
        let mut ll = Vec::with_capacity(n);
        for _ in 0..n {
            let i = rng.random_range(0..n);
            ss.push(scores[i]);
            ll.push(labels[i]);
        }
        if let Some(auc) = roc_auc(&ss, &ll) {
            aucs.push(auc);
        }
    }
    if aucs.len() < resamples / 2 {
        return None;
    }
    aucs.sort_by(f64::total_cmp);
    let lo = aucs[((aucs.len() as f64 * 0.025) as usize).min(aucs.len() - 1)];
    let hi = aucs[((aucs.len() as f64 * 0.975) as usize).min(aucs.len() - 1)];
    Some((lo, hi))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auc_known_values() {
        let scores = [0.1, 0.2, 0.8, 0.9];
        let labels = [false, false, true, true];
        assert!((roc_auc(&scores, &labels).unwrap() - 1.0).abs() < 1e-12);
        assert!((roc_auc(&scores, &[true, true, false, false]).unwrap() - 0.0).abs() < 1e-12);
        assert!((roc_auc(&scores, &[false, true, false, true]).unwrap() - 0.75).abs() < 1e-12);
    }

    #[test]
    fn auc_ties_count_half() {
        let scores = [0.5, 0.5];
        assert!((roc_auc(&scores, &[true, false]).unwrap() - 0.5).abs() < 1e-12);

        let scores = [0.5, 0.5, 0.9];
        let labels = [true, false, true];
        // one positive outranks the negative, one ties: (1 + 0.5) / 2 = 0.75
        assert!((roc_auc(&scores, &labels).unwrap() - 0.75).abs() < 1e-12);
    }

    #[test]
    fn auc_matches_bruteforce() {
        let scores = [0.3, -0.2, 1.0, 1.0, 0.0, -0.7, 0.4];
        let labels = [true, false, true, false, true, false, true];
        let got = roc_auc(&scores, &labels).unwrap();
        let mut pos = Vec::new();
        let mut neg = Vec::new();
        for (&s, &y) in scores.iter().zip(&labels) {
            if y {
                pos.push(s);
            } else {
                neg.push(s);
            }
        }
        let mut wins = 0.0;
        for &p in &pos {
            for &n in &neg {
                if p > n {
                    wins += 1.0;
                } else if p == n {
                    wins += 0.5;
                }
            }
        }
        let want = wins / (pos.len() * neg.len()) as f64;
        assert!((got - want).abs() < 1e-12);
    }

    #[test]
    fn balanced_accuracy_perfect_separation() {
        let scores = [-0.9, -0.4, 0.6, 0.8];
        let labels = [false, false, true, true];
        let (ba, t) = best_balanced_accuracy(&scores, &labels).unwrap();
        assert!((ba - 1.0).abs() < 1e-12);
        assert!(t > -0.4 && t < 0.6);
    }

    #[test]
    fn cohens_d_sign_and_magnitude() {
        let scores = [-1.0, -1.1, -0.9, 1.0, 1.1, 0.9];
        let labels = [false, false, false, true, true, true];
        let d = cohens_d(&scores, &labels).unwrap();
        assert!(d > 10.0, "d = {d}");
    }

    #[test]
    fn platt_learns_direction() {
        let scores = [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0];
        let labels = [false, false, false, false, true, true, true, true];
        let p = fit_platt(&scores, &labels).unwrap();
        assert!(p.probability(2.0) > 0.8, "p = {}", p.probability(2.0));
        assert!(p.probability(-2.0) < 0.2, "p = {}", p.probability(-2.0));
    }

    #[test]
    fn bootstrap_ci_contains_auc() {
        let scores: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let labels: Vec<bool> = (0..40).map(|i| i >= 20).collect();
        let (lo, hi) = bootstrap_auc_ci(&scores, &labels, 7, 200).unwrap();
        assert!(lo <= 1.0 && hi <= 1.0);
        assert!((lo - 1.0).abs() < 1e-9, "lo = {lo}");
    }
}
