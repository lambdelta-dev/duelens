//! Exemplar-pair search.
//!
//! For a pair `(g, b)` the surrogate score of an image `x` is
//! `s(x) = d(x, b) - d(x, g)`. The pair is evaluated on all images except `g`
//! and `b` (leave-two-out), so exemplars do not get to vote for themselves.
//! Search is exhaustive for small candidate spaces and beam search otherwise.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::ValueEnum;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use serde::Serialize;

use crate::dataset::{self, Dataset, DatasetSpec, Skipped};
use crate::features;
use crate::metrics;

pub const TOOL_NAME: &str = "duelens";

/// Metric used to rank exemplar pairs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Objective {
    /// Threshold-free probability of correct ordering (default).
    Auc,
    /// Balanced accuracy at the best threshold.
    Balanced,
    /// Cohen's d effect size between score distributions.
    Effect,
    /// 0.5 * AUC + 0.3 * balanced accuracy + 0.2 * sigmoid(Cohen's d).
    Blend,
}

impl Objective {
    pub fn as_str(self) -> &'static str {
        match self {
            Objective::Auc => "auc",
            Objective::Balanced => "balanced",
            Objective::Effect => "effect",
            Objective::Blend => "blend",
        }
    }

    fn score(self, auc: f64, balanced: f64, d: Option<f64>) -> f64 {
        let d = d.unwrap_or(0.0);
        match self {
            Objective::Auc => auc,
            Objective::Balanced => balanced,
            Objective::Effect => d,
            Objective::Blend => 0.5 * auc + 0.3 * balanced + 0.2 * sigmoid(d),
        }
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Symmetric matrix of Euclidean distances between descriptors.
pub struct DistanceMatrix {
    n: usize,
    data: Vec<f32>,
}

impl DistanceMatrix {
    pub fn build(features: &[Vec<f32>]) -> Self {
        let n = features.len();
        let rows: Vec<Vec<f32>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut row = vec![0f32; n];
                for j in 0..=i {
                    row[j] = features::sq_distance(&features[i], &features[j]).sqrt();
                }
                row
            })
            .collect();
        let mut data = vec![0f32; n * n];
        for (i, row) in rows.iter().enumerate() {
            data[i * n..(i + 1) * n].copy_from_slice(row);
            for j in 0..i {
                data[j * n + i] = row[j];
            }
        }
        Self { n, data }
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.n + j] as f64
    }

    pub fn n(&self) -> usize {
        self.n
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PairScore {
    pub good: String,
    pub bad: String,
    pub auc: f64,
    pub balanced_accuracy: f64,
    pub threshold: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cohens_d: Option<f64>,
    pub objective: f64,
    pub held_out_good: usize,
    pub held_out_bad: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auc_ci95: Option<[f64; 2]>,
    pub auc_with_exemplars: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageScore {
    pub path: String,
    pub label: &'static str,
    pub held_out: bool,
    pub d_good: f64,
    pub d_bad: f64,
    pub score: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_good: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CalibrationInfo {
    pub slope: f64,
    pub intercept: f64,
    pub threshold: f64,
    pub threshold_probability: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct DatasetSummary {
    pub root: String,
    pub good_label: String,
    pub bad_label: String,
    pub manifest: Option<String>,
    pub recursive: bool,
    pub n_good: usize,
    pub n_bad: usize,
    pub n_total: usize,
    pub skipped: Vec<Skipped>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchSummary {
    pub mode: String,
    pub pairs_possible: usize,
    pub pairs_evaluated: usize,
    pub auc_best: f64,
    pub auc_median: f64,
    pub auc_worst: f64,
    pub features: String,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RunConfig {
    pub objective: String,
    pub top_k: usize,
    pub seed: u64,
    pub max_exhaustive: usize,
    pub beam_width: usize,
    pub leave_two_out: bool,
    pub bootstrap_resamples: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct FindReport {
    pub tool: String,
    pub version: String,
    pub config: RunConfig,
    pub dataset: DatasetSummary,
    pub search: SearchSummary,
    pub winner: Option<PairScore>,
    pub top: Vec<PairScore>,
    pub winner_scores: Vec<ImageScore>,
    pub calibration: Option<CalibrationInfo>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct FindOptions {
    pub root: PathBuf,
    pub good_label: String,
    pub bad_label: String,
    pub manifest: Option<PathBuf>,
    pub recursive: bool,
    pub objective: Objective,
    pub top_k: usize,
    pub seed: u64,
    pub max_exhaustive: usize,
    pub beam_width: usize,
    pub bootstrap_resamples: usize,
    pub verbose: bool,
}

impl Default for FindOptions {
    fn default() -> Self {
        Self {
            root: PathBuf::from("."),
            good_label: "good".to_string(),
            bad_label: "bad".to_string(),
            manifest: None,
            recursive: false,
            objective: Objective::Auc,
            top_k: 10,
            seed: 42,
            max_exhaustive: 200_000,
            beam_width: 16,
            bootstrap_resamples: 400,
            verbose: true,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Candidate {
    pub good: usize,
    pub bad: usize,
    pub auc: f64,
    pub balanced_accuracy: f64,
    pub threshold: f64,
    pub cohens_d: Option<f64>,
    pub objective: f64,
    pub held_out_good: usize,
    pub held_out_bad: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    Exhaustive,
    Beam,
}

impl SearchMode {
    pub fn as_str(self) -> &'static str {
        match self {
            SearchMode::Exhaustive => "exhaustive",
            SearchMode::Beam => "beam",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchConfig {
    pub objective: Objective,
    pub max_exhaustive: usize,
    pub beam_width: usize,
    pub seed: u64,
    pub max_rounds: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            objective: Objective::Auc,
            max_exhaustive: 200_000,
            beam_width: 16,
            seed: 42,
            max_rounds: 20,
        }
    }
}

pub struct Selection {
    pub candidates: Vec<Candidate>,
    pub mode: SearchMode,
    pub evaluated: usize,
}

/// Run the full pipeline: load, describe, search, score, report.
pub fn run_find(options: &FindOptions) -> Result<FindReport> {
    let start = Instant::now();
    if options.verbose {
        eprintln!("duelens: scanning {}", options.root.display());
    }
    let spec = DatasetSpec {
        root: options.root.clone(),
        good_label: options.good_label.clone(),
        bad_label: options.bad_label.clone(),
        manifest: options.manifest.clone(),
        recursive: options.recursive,
    };
    let dataset = dataset::load(&spec)?;
    if options.verbose {
        eprintln!(
            "duelens: {} images ({} good / {} bad){}",
            dataset.n(),
            dataset.good.len(),
            dataset.bad.len(),
            if dataset.skipped.is_empty() {
                String::new()
            } else {
                format!(", {} skipped", dataset.skipped.len())
            }
        );
    }

    let features: Vec<Vec<f32>> = dataset.samples.iter().map(|s| s.features.clone()).collect();
    let distances = DistanceMatrix::build(&features);
    let labels = dataset.labels();
    let total_pairs = dataset.good.len() * dataset.bad.len();
    let mode = if total_pairs <= options.max_exhaustive {
        SearchMode::Exhaustive
    } else {
        SearchMode::Beam
    };
    if options.verbose {
        if mode == SearchMode::Exhaustive {
            eprintln!("duelens: scoring all {total_pairs} pairs...");
        } else {
            eprintln!(
                "duelens: scoring {total_pairs} possible pairs with beam search (width {})...",
                options.beam_width.max(1)
            );
        }
    }

    let config = SearchConfig {
        objective: options.objective,
        max_exhaustive: options.max_exhaustive,
        beam_width: options.beam_width.max(1),
        seed: options.seed,
        max_rounds: 20,
    };
    let selection = select_pairs(&distances, &labels, &dataset.good, &dataset.bad, &config);
    if selection.candidates.is_empty() {
        anyhow::bail!("no candidate pairs could be scored");
    }

    let mut aucs: Vec<f64> = selection.candidates.iter().map(|c| c.auc).collect();
    aucs.sort_by(f64::total_cmp);
    let auc_best = aucs.last().copied().unwrap_or(0.0);
    let auc_worst = aucs.first().copied().unwrap_or(0.0);
    let auc_median = aucs[aucs.len() / 2];

    let top_k = options.top_k.max(1).min(selection.candidates.len());
    let mut top = Vec::with_capacity(top_k);
    for candidate in selection.candidates.iter().take(top_k) {
        top.push(to_pair_score(
            &dataset, &distances, &labels, candidate, None,
        ));
    }

    let winner_candidate = selection.candidates[0];
    let winner_ci = if options.bootstrap_resamples > 0 {
        let (scores, ys) = pair_scores(
            &distances,
            &labels,
            winner_candidate.good,
            winner_candidate.bad,
            true,
        );
        metrics::bootstrap_auc_ci(
            &scores,
            &ys,
            options.seed ^ 0x9E37_79B9_7F4A_7C15,
            options.bootstrap_resamples,
        )
        .map(|(lo, hi)| [lo, hi])
    } else {
        None
    };
    if let Some(first) = top.first_mut() {
        first.auc_ci95 = winner_ci;
    }

    let mut calibration = None;
    let mut winner_scores = Vec::with_capacity(dataset.n());
    {
        let (held_scores, held_labels) = pair_scores(
            &distances,
            &labels,
            winner_candidate.good,
            winner_candidate.bad,
            true,
        );
        let platt = metrics::fit_platt(&held_scores, &held_labels);
        if let Some(platt) = platt {
            calibration = Some(CalibrationInfo {
                slope: platt.slope,
                intercept: platt.intercept,
                threshold: winner_candidate.threshold,
                threshold_probability: platt.probability(winner_candidate.threshold),
            });
        }
        for (idx, sample) in dataset.samples.iter().enumerate() {
            let d_good = distances.get(idx, winner_candidate.good);
            let d_bad = distances.get(idx, winner_candidate.bad);
            let score = d_bad - d_good;
            winner_scores.push(ImageScore {
                path: sample.path.display().to_string(),
                label: sample.label.as_str(),
                held_out: idx != winner_candidate.good && idx != winner_candidate.bad,
                d_good,
                d_bad,
                score,
                p_good: platt.map(|p| p.probability(score)),
            });
        }
    }

    let mut notes = Vec::new();
    if mode == SearchMode::Beam {
        notes.push(format!(
            "Candidate space had {total_pairs} pairs (> --max-exhaustive {}); beam search is approximate, not guaranteed optimal.",
            options.max_exhaustive
        ));
    }
    if !dataset.skipped.is_empty() {
        notes.push(format!(
            "{} files were skipped (undecodable or unlabelled); see dataset.skipped.",
            dataset.skipped.len()
        ));
    }
    notes.push(
        "Scores use a classical-CV surrogate descriptor, not your decision API. Validate the shortlist against the real API before shipping."
            .to_string(),
    );

    let elapsed = start.elapsed().as_millis() as u64;
    Ok(FindReport {
        tool: TOOL_NAME.to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        config: RunConfig {
            objective: options.objective.as_str().to_string(),
            top_k,
            seed: options.seed,
            max_exhaustive: options.max_exhaustive,
            beam_width: options.beam_width,
            leave_two_out: true,
            bootstrap_resamples: options.bootstrap_resamples,
        },
        dataset: DatasetSummary {
            root: dataset.root.display().to_string(),
            good_label: options.good_label.clone(),
            bad_label: options.bad_label.clone(),
            manifest: options.manifest.as_ref().map(|p| p.display().to_string()),
            recursive: options.recursive,
            n_good: dataset.good.len(),
            n_bad: dataset.bad.len(),
            n_total: dataset.n(),
            skipped: dataset.skipped,
        },
        search: SearchSummary {
            mode: mode.as_str().to_string(),
            pairs_possible: total_pairs,
            pairs_evaluated: selection.evaluated,
            auc_best,
            auc_median,
            auc_worst,
            features: features::FEATURE_VERSION.to_string(),
            elapsed_ms: elapsed,
        },
        winner: top.first().cloned(),
        top,
        winner_scores,
        calibration,
        notes,
    })
}

/// Search all candidate pairs, returning them sorted best-first.
pub fn select_pairs(
    distances: &DistanceMatrix,
    labels: &[bool],
    good: &[usize],
    bad: &[usize],
    config: &SearchConfig,
) -> Selection {
    let total = good.len() * bad.len();
    if total <= config.max_exhaustive {
        let pairs: Vec<(usize, usize)> = good
            .iter()
            .flat_map(|&g| bad.iter().map(move |&b| (g, b)))
            .collect();
        let mut candidates: Vec<Candidate> = pairs
            .par_iter()
            .map(|&(g, b)| evaluate(distances, labels, g, b, config.objective))
            .collect();
        candidates.sort_by(candidate_order);
        Selection {
            candidates,
            mode: SearchMode::Exhaustive,
            evaluated: total,
        }
    } else {
        let (mut candidates, evaluated) = beam_search(distances, labels, good, bad, config);
        candidates.sort_by(candidate_order);
        Selection {
            candidates,
            mode: SearchMode::Beam,
            evaluated,
        }
    }
}

fn beam_search(
    distances: &DistanceMatrix,
    labels: &[bool],
    good: &[usize],
    bad: &[usize],
    config: &SearchConfig,
) -> (Vec<Candidate>, usize) {
    let width = config.beam_width.max(1);
    let total = good.len() * bad.len();
    let mut rng = SmallRng::seed_from_u64(config.seed);
    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    let mut seeds: Vec<(usize, usize)> = Vec::new();

    if let (Some(mg), Some(mb)) = (medoid(distances, good), medoid(distances, bad)) {
        seeds.push((mg, mb));
        seen.insert((mg, mb));
    }
    let seed_target = width.saturating_mul(8).clamp(64, 4096).min(total);
    while seeds.len() < seed_target {
        let g = good[rng.random_range(0..good.len())];
        let b = bad[rng.random_range(0..bad.len())];
        if seen.insert((g, b)) {
            seeds.push((g, b));
        }
    }

    let mut evaluated = seeds.len();
    let mut candidates: Vec<Candidate> = seeds
        .par_iter()
        .map(|&(g, b)| evaluate(distances, labels, g, b, config.objective))
        .collect();
    candidates.sort_by(candidate_order);
    candidates.truncate(width);
    let mut best = candidates[0].objective;
    let mut stale = 0usize;

    for _ in 0..config.max_rounds.max(1) {
        let mut around = Vec::with_capacity(candidates.len() * (good.len() + bad.len()));
        for candidate in &candidates {
            for &g in good {
                around.push((g, candidate.bad));
            }
            for &b in bad {
                around.push((candidate.good, b));
            }
        }
        let mut fresh = Vec::new();
        for pair in around {
            if seen.insert(pair) {
                fresh.push(pair);
            }
        }
        if fresh.is_empty() {
            break;
        }
        evaluated += fresh.len();
        let mut new_candidates: Vec<Candidate> = fresh
            .par_iter()
            .map(|&(g, b)| evaluate(distances, labels, g, b, config.objective))
            .collect();
        candidates.append(&mut new_candidates);
        candidates.sort_by(candidate_order);
        candidates.truncate(width);

        if candidates[0].objective > best + 1e-9 {
            best = candidates[0].objective;
            stale = 0;
        } else {
            stale += 1;
            if stale >= 2 {
                break;
            }
        }
    }

    (candidates, evaluated)
}

fn medoid(distances: &DistanceMatrix, class: &[usize]) -> Option<usize> {
    class.iter().copied().min_by(|&a, &b| {
        let da: f64 = class.iter().map(|&x| distances.get(a, x)).sum();
        let db: f64 = class.iter().map(|&x| distances.get(b, x)).sum();
        da.total_cmp(&db)
    })
}

fn evaluate(
    distances: &DistanceMatrix,
    labels: &[bool],
    good: usize,
    bad: usize,
    objective: Objective,
) -> Candidate {
    let (scores, ys) = pair_scores(distances, labels, good, bad, true);
    let auc = metrics::roc_auc(&scores, &ys).unwrap_or(0.5);
    let (balanced_accuracy, threshold) =
        metrics::best_balanced_accuracy(&scores, &ys).unwrap_or((0.5, 0.0));
    let cohens_d = metrics::cohens_d(&scores, &ys);
    let n_good = ys.iter().filter(|&&y| y).count();
    let n_bad = ys.len() - n_good;
    Candidate {
        good,
        bad,
        auc,
        balanced_accuracy,
        threshold,
        cohens_d,
        objective: objective.score(auc, balanced_accuracy, cohens_d),
        held_out_good: n_good,
        held_out_bad: n_bad,
    }
}

/// Return `(scores, is_good)` for all images, optionally excluding the two
/// exemplars from the evaluation.
fn pair_scores(
    distances: &DistanceMatrix,
    labels: &[bool],
    good: usize,
    bad: usize,
    held_out_only: bool,
) -> (Vec<f64>, Vec<bool>) {
    let mut scores = Vec::with_capacity(labels.len());
    let mut ys = Vec::with_capacity(labels.len());
    for (x, &is_good) in labels.iter().enumerate() {
        if held_out_only && (x == good || x == bad) {
            continue;
        }
        scores.push(distances.get(x, bad) - distances.get(x, good));
        ys.push(is_good);
    }
    (scores, ys)
}

fn to_pair_score(
    dataset: &Dataset,
    distances: &DistanceMatrix,
    labels: &[bool],
    candidate: &Candidate,
    auc_ci95: Option<[f64; 2]>,
) -> PairScore {
    let (all_scores, all_labels) =
        pair_scores(distances, labels, candidate.good, candidate.bad, false);
    let auc_all = metrics::roc_auc(&all_scores, &all_labels).unwrap_or(0.5);
    PairScore {
        good: dataset.samples[candidate.good].path.display().to_string(),
        bad: dataset.samples[candidate.bad].path.display().to_string(),
        auc: candidate.auc,
        balanced_accuracy: candidate.balanced_accuracy,
        threshold: candidate.threshold,
        cohens_d: candidate.cohens_d,
        objective: candidate.objective,
        held_out_good: candidate.held_out_good,
        held_out_bad: candidate.held_out_bad,
        auc_ci95,
        auc_with_exemplars: auc_all,
    }
}

fn candidate_order(a: &Candidate, b: &Candidate) -> Ordering {
    b.objective
        .total_cmp(&a.objective)
        .then_with(|| b.auc.total_cmp(&a.auc))
        .then_with(|| b.balanced_accuracy.total_cmp(&a.balanced_accuracy))
        .then_with(|| {
            b.cohens_d
                .unwrap_or(f64::NEG_INFINITY)
                .total_cmp(&a.cohens_d.unwrap_or(f64::NEG_INFINITY))
        })
        .then_with(|| a.good.cmp(&b.good))
        .then_with(|| a.bad.cmp(&b.bad))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dataset_features() -> (Vec<Vec<f32>>, Vec<bool>, Vec<usize>, Vec<usize>) {
        // Two clusters with one good outlier. The best pair should be a
        // prototypical good and a prototypical bad image.
        let features = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.2, -0.1],
            vec![2.0, 2.0], // good outlier
            vec![5.0, 5.0],
            vec![5.2, 4.8],
            vec![4.8, 5.3],
        ];
        let labels = vec![true, true, true, true, false, false, false];
        (features, labels, vec![0, 1, 2, 3], vec![4, 5, 6])
    }

    #[test]
    fn exhaustive_scores_every_pair_and_ranks_by_auc() {
        let (features, labels, good, bad) = dataset_features();
        let dm = DistanceMatrix::build(&features);
        let config = SearchConfig {
            objective: Objective::Auc,
            max_exhaustive: 1000,
            ..Default::default()
        };
        let selection = select_pairs(&dm, &labels, &good, &bad, &config);
        assert_eq!(selection.mode, SearchMode::Exhaustive);
        assert_eq!(selection.evaluated, 12);
        assert_eq!(selection.candidates.len(), 12);
        let best = selection.candidates[0];
        assert!(best.auc >= 0.99, "best auc = {}", best.auc);
        assert!(good.contains(&best.good));
        assert!(bad.contains(&best.bad));
        for pair in selection.candidates.windows(2) {
            assert!(pair[0].objective >= pair[1].objective);
        }
    }

    #[test]
    fn beam_search_finds_a_good_pair_and_is_deterministic() {
        let (features, labels, good, bad) = dataset_features();
        let dm = DistanceMatrix::build(&features);
        let config = SearchConfig {
            objective: Objective::Auc,
            max_exhaustive: 3,
            beam_width: 3,
            seed: 9,
            max_rounds: 10,
        };
        let a = select_pairs(&dm, &labels, &good, &bad, &config);
        let b = select_pairs(&dm, &labels, &good, &bad, &config);
        assert_eq!(a.mode, SearchMode::Beam);
        assert!(a.evaluated <= 12);
        assert!(a.candidates[0].auc > 0.9);
        assert_eq!(
            (a.candidates[0].good, a.candidates[0].bad),
            (b.candidates[0].good, b.candidates[0].bad)
        );
        assert!(a.candidates.len() <= 3);
    }

    #[test]
    fn beam_search_expands_beyond_its_seeds() {
        // Two well-separated 2-D clusters; pair space is much larger than the
        // beam, so the search must expand around its seeds.
        let mut features = Vec::new();
        for i in 0..20 {
            let jitter = (i as f32 % 5.0) * 0.01;
            features.push(vec![0.0 + jitter, 0.0 - jitter]);
        }
        for i in 0..20 {
            let jitter = (i as f32 % 7.0) * 0.01;
            features.push(vec![10.0 + jitter, 10.0 + jitter]);
        }
        let labels: Vec<bool> = (0..40).map(|i| i < 20).collect();
        let good: Vec<usize> = (0..20).collect();
        let bad: Vec<usize> = (20..40).collect();
        let dm = DistanceMatrix::build(&features);
        let config = SearchConfig {
            objective: Objective::Auc,
            max_exhaustive: 5,
            beam_width: 4,
            seed: 3,
            max_rounds: 12,
        };
        let selection = select_pairs(&dm, &labels, &good, &bad, &config);
        assert_eq!(selection.mode, SearchMode::Beam);
        assert!(selection.evaluated > 32);
        assert!(selection.evaluated < 400);
        assert!(selection.candidates.len() <= 4);
        assert!(selection.candidates[0].auc > 0.99);
    }

    #[test]
    fn leave_two_out_excludes_exemplars() {
        let (features, labels, _, _) = dataset_features();
        let dm = DistanceMatrix::build(&features);
        let (scores, ys) = pair_scores(&dm, &labels, 0, 4, true);
        assert_eq!(scores.len(), labels.len() - 2);
        assert_eq!(ys.len(), labels.len() - 2);
    }
}
