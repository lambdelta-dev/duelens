//! Terminal, Markdown, JSON and image export rendering.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde_json::json;

use crate::search::{FindReport, ImageScore};

/// Render the short human-readable summary printed to stdout.
pub fn render_terminal(report: &FindReport) -> String {
    let mut out = String::new();
    let ds = &report.dataset;
    let search = &report.search;
    let _ = writeln!(
        out,
        "duelens {}  ·  {}  ·  {} objective",
        report.version, search.features, report.config.objective
    );
    let _ = writeln!(
        out,
        "{} good / {} bad  ·  {} · {} pairs evaluated · {}",
        ds.n_good,
        ds.n_bad,
        search.mode,
        search.pairs_evaluated,
        human_duration(search.elapsed_ms)
    );
    if search.mode == "exhaustive" {
        let _ = writeln!(
            out,
            "pair-choice effect: AUC {:.3} (worst) · {:.3} (median) · {:.3} (best) over all pairs",
            search.auc_worst, search.auc_median, search.auc_best
        );
    } else {
        let _ = writeln!(
            out,
            "beam mode: {} pairs evaluated; final-beam AUC {:.3}–{:.3} (approximate search)",
            search.pairs_evaluated, search.auc_worst, search.auc_best
        );
    }
    if !ds.skipped.is_empty() {
        let _ = writeln!(out, "{} files skipped", ds.skipped.len());
    }

    if let Some(winner) = &report.winner {
        let _ = writeln!(out, "\nWINNER");
        let _ = writeln!(out, "  good   {}", winner.good);
        let _ = writeln!(out, "  bad    {}", winner.bad);
        let ci = winner
            .auc_ci95
            .map(|[lo, hi]| format!("  [{lo:.3}, {hi:.3}] 95% CI"))
            .unwrap_or_default();
        let _ = writeln!(out, "  AUC    {:.3}{ci}", winner.auc);
        let d = winner
            .cohens_d
            .map(|d| format!("{d:.3}"))
            .unwrap_or_else(|| "n/a".to_string());
        let _ = writeln!(
            out,
            "  bal. acc {:.3}   Cohen's d {}   threshold {:.4}",
            winner.balanced_accuracy, d, winner.threshold
        );
        if let Some(cal) = &report.calibration {
            let _ = writeln!(
                out,
                "  P(good) at threshold {:.3}   AUC incl. exemplars {:.3}",
                cal.threshold_probability, winner.auc_with_exemplars
            );
        } else {
            let _ = writeln!(
                out,
                "  AUC incl. exemplars {:.3}",
                winner.auc_with_exemplars
            );
        }

        let _ = writeln!(out, "\nTOP PAIRS");
        let _ = writeln!(
            out,
            "  {:>2}  {:>5}  {:>5}  {:>6}  {:<36}  bad",
            "#", "AUC", "BAL", "d", "good"
        );
        for (i, pair) in report.top.iter().enumerate() {
            let _ = writeln!(
                out,
                "  {:>2}  {:>5.3}  {:>5.3}  {:>6}  {:<36}  {}",
                i + 1,
                pair.auc,
                pair.balanced_accuracy,
                pair.cohens_d
                    .map(|d| format!("{d:.3}"))
                    .unwrap_or_else(|| "n/a".to_string()),
                ellipsis_tail(&pair.good, 36),
                ellipsis_tail(&pair.bad, 42)
            );
        }

        if report.winner_scores.iter().any(|s| s.held_out) {
            let _ = writeln!(
                out,
                "\nHELD-OUT SCORE DISTRIBUTION  (▓ good, ░ bad; s = d_bad - d_good)"
            );
            let _ = write!(out, "{}", histogram(&report.winner_scores, 32, 8));
            let _ = writeln!(out, "  threshold {:.4}", winner.threshold);
        }
    } else {
        let _ = writeln!(out, "\nno winner");
    }

    if !report.notes.is_empty() {
        let _ = writeln!(out, "\nNOTES");
        for note in &report.notes {
            let _ = writeln!(out, "  - {note}");
        }
    }
    out
}

/// Render a self-contained Markdown report.
pub fn render_markdown(report: &FindReport) -> String {
    let mut out = String::new();
    let ds = &report.dataset;
    let search = &report.search;
    let _ = writeln!(out, "# duelens report\n");
    let _ = writeln!(
        out,
        "`duelens` v{} · descriptor `{}` · objective `{}`\n",
        report.version, search.features, report.config.objective
    );
    let _ = writeln!(
        out,
        "Dataset: **{} good / {} bad** images from `{}` · search: `{}` · {} pairs evaluated in {}\n",
        ds.n_good,
        ds.n_bad,
        escape_md(&ds.root),
        search.mode,
        search.pairs_evaluated,
        human_duration(search.elapsed_ms)
    );
    if search.mode == "exhaustive" {
        let _ = writeln!(
            out,
            "AUC across all candidate pairs: **{:.4}** worst, **{:.4}** median, **{:.4}** best.\n",
            search.auc_worst, search.auc_median, search.auc_best
        );
    } else {
        let _ = writeln!(
            out,
            "Beam search evaluated {} of {} pairs; the final beam spans AUC {:.4}–{:.4}. The search is approximate.\n",
            search.pairs_evaluated, search.pairs_possible, search.auc_worst, search.auc_best
        );
    }

    if let Some(winner) = &report.winner {
        let _ = writeln!(out, "## Winner\n");
        let _ = writeln!(out, "| role | image |\n|---|---|");
        let _ = writeln!(out, "| good exemplar | `{}` |", escape_md(&winner.good));
        let _ = writeln!(out, "| bad exemplar | `{}` |\n", escape_md(&winner.bad));
        let _ = writeln!(out, "| metric | value |\n|---|---|");
        let _ = writeln!(out, "| AUC (leave-two-out) | {:.4} |", winner.auc);
        if let Some([lo, hi]) = winner.auc_ci95 {
            let _ = writeln!(out, "| AUC 95% bootstrap CI | [{lo:.4}, {hi:.4}] |");
        }
        let _ = writeln!(
            out,
            "| balanced accuracy | {:.4} |",
            winner.balanced_accuracy
        );
        let d = winner
            .cohens_d
            .map(|d| format!("{d:.4}"))
            .unwrap_or_else(|| "n/a".to_string());
        let _ = writeln!(out, "| Cohen's d | {d} |");
        let _ = writeln!(
            out,
            "| decision threshold `s > t` | {:.4} |",
            winner.threshold
        );
        let _ = writeln!(
            out,
            "| AUC including the two exemplars | {:.4} |",
            winner.auc_with_exemplars
        );
        if let Some(cal) = &report.calibration {
            let _ = writeln!(
                out,
                "| calibrated P(good) at threshold | {:.4} |",
                cal.threshold_probability
            );
        }
        let _ = writeln!(out);

        let _ = writeln!(out, "## Top pairs\n");
        let _ = writeln!(
            out,
            "| # | AUC | balanced | Cohen's d | good exemplar | bad exemplar |\n|---:|---:|---:|---:|---|---|"
        );
        for (i, pair) in report.top.iter().enumerate() {
            let _ = writeln!(
                out,
                "| {} | {:.4} | {:.4} | {} | `{}` | `{}` |",
                i + 1,
                pair.auc,
                pair.balanced_accuracy,
                pair.cohens_d
                    .map(|d| format!("{d:.4}"))
                    .unwrap_or_else(|| "n/a".to_string()),
                escape_md(&pair.good),
                escape_md(&pair.bad)
            );
        }
        let _ = writeln!(out);

        let _ = writeln!(out, "## Score distribution\n");
        let _ = writeln!(
            out,
            "Every held-out image is scored with `s = d(x, bad) - d(x, good)`; larger is more good-like."
        );
        let _ = writeln!(out, "```text");
        let _ = write!(out, "{}", histogram(&report.winner_scores, 40, 10));
        let _ = writeln!(out, "threshold s > {:.4}", winner.threshold);
        let _ = writeln!(out, "```\n");

        if let Some(cal) = &report.calibration {
            let _ = writeln!(out, "## Calibration\n");
            let _ = writeln!(
                out,
                "`P(good | s) = sigmoid({:.4} * s + {:.4})`, fitted with lightly regularised logistic regression on the held-out scores. The sign is data-driven; a positive slope means larger `s` means more good-like.\n",
                cal.slope, cal.intercept
            );
        }

        let mut held: Vec<&ImageScore> =
            report.winner_scores.iter().filter(|s| s.held_out).collect();
        held.sort_by(|a, b| b.score.total_cmp(&a.score));
        let _ = writeln!(out, "## Held-out images\n");
        let _ = writeln!(
            out,
            "| image | label | d(good) | d(bad) | s | P(good) |\n|---|---|---:|---:|---:|---:|"
        );
        for image in held {
            let p = image
                .p_good
                .map(|p| format!("{p:.3}"))
                .unwrap_or_else(|| "—".to_string());
            let _ = writeln!(
                out,
                "| `{}` | {} | {:.4} | {:.4} | {:.4} | {} |",
                escape_md(&image.path),
                image.label,
                image.d_good,
                image.d_bad,
                image.score,
                p
            );
        }
        let _ = writeln!(out);
    }

    if !report.notes.is_empty() {
        let _ = writeln!(out, "## Notes\n");
        for note in &report.notes {
            let _ = writeln!(out, "- {note}");
        }
        let _ = writeln!(out);
    }

    let _ = writeln!(out, "## Reproduce\n");
    let _ = writeln!(out, "```sh");
    let _ = writeln!(out, "{}", reproduce_command(report));
    let _ = writeln!(out, "```");
    out
}

/// Write pretty JSON.
pub fn write_json(report: &FindReport, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("cannot create {}", parent.display()))?;
    }
    let bytes = serde_json::to_vec_pretty(report)
        .with_context(|| format!("cannot serialise report for {}", path.display()))?;
    fs::write(path, bytes).with_context(|| format!("cannot write {}", path.display()))?;
    Ok(())
}

/// Write the Markdown report.
pub fn write_markdown(report: &FindReport, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("cannot create {}", parent.display()))?;
    }
    fs::write(path, render_markdown(report))
        .with_context(|| format!("cannot write {}", path.display()))?;
    Ok(())
}

/// Copy the winning pair into `dir` as `good.<ext>`, `bad.<ext>` plus
/// `pair.json` describing the selection.
pub fn export_winner(report: &FindReport, dir: &Path) -> Result<Vec<PathBuf>> {
    let Some(winner) = &report.winner else {
        bail!("no winner to export");
    };
    fs::create_dir_all(dir)
        .with_context(|| format!("cannot create export directory {}", dir.display()))?;
    let mut written = Vec::new();
    for (src, stem) in [(&winner.good, "good"), (&winner.bad, "bad")] {
        let src_path = Path::new(src);
        let ext = src_path
            .extension()
            .map(|e| e.to_string_lossy().to_string())
            .unwrap_or_else(|| "png".to_string());
        let dest = dir.join(format!("{stem}.{ext}"));
        let src_canonical = fs::canonicalize(src_path).ok();
        let dest_canonical = fs::canonicalize(&dest).ok();
        if src_canonical.is_some() && src_canonical == dest_canonical {
            bail!(
                "refusing to copy {} onto itself; pick a different --export directory",
                src_path.display()
            );
        }
        fs::copy(src_path, &dest).with_context(|| {
            format!(
                "cannot copy winning image {} to {}",
                src_path.display(),
                dest.display()
            )
        })?;
        written.push(dest);
    }
    let meta_path = dir.join("pair.json");
    let meta = json!({
        "tool": report.tool,
        "version": report.version,
        "objective": report.config.objective,
        "winner": winner,
        "dataset": {
            "n_good": report.dataset.n_good,
            "n_bad": report.dataset.n_bad,
            "root": report.dataset.root,
            "good_label": report.dataset.good_label,
            "bad_label": report.dataset.bad_label,
        }
    });
    fs::write(&meta_path, serde_json::to_vec_pretty(&meta)?)
        .with_context(|| format!("cannot write {}", meta_path.display()))?;
    written.push(meta_path);
    Ok(written)
}

/// A compact two-row histogram over the held-out scores. `▓` is a good image,
/// `░` is a bad image; each bin has one character per class.
pub fn histogram(scores: &[ImageScore], bins: usize, height: usize) -> String {
    let held: Vec<&ImageScore> = scores.iter().filter(|s| s.held_out).collect();
    if held.is_empty() || bins == 0 || height == 0 {
        return "  (no held-out images)\n".to_string();
    }
    let min = held.iter().map(|s| s.score).fold(f64::INFINITY, f64::min);
    let max = held
        .iter()
        .map(|s| s.score)
        .fold(f64::NEG_INFINITY, f64::max);
    let (min, max) = if (max - min).abs() < 1e-9 {
        (min - 0.5, max + 0.5)
    } else {
        (min, max)
    };
    let mut good = vec![0usize; bins];
    let mut bad = vec![0usize; bins];
    for image in held {
        let t = ((image.score - min) / (max - min) * (bins as f64 - 1e-9)) as usize;
        let bin = t.min(bins - 1);
        if image.label == "good" {
            good[bin] += 1;
        } else {
            bad[bin] += 1;
        }
    }
    let peak = good.iter().chain(&bad).copied().max().unwrap_or(0);
    let heights: Vec<(usize, usize)> = good
        .iter()
        .zip(&bad)
        .map(|(&g, &b)| {
            (
                if peak == 0 {
                    0
                } else {
                    (g * height).div_ceil(peak)
                },
                if peak == 0 {
                    0
                } else {
                    (b * height).div_ceil(peak)
                },
            )
        })
        .collect();

    let mut out = String::new();
    let width = bins * 2;
    for level in (1..=height).rev() {
        let mut line = String::with_capacity(width + 4);
        line.push_str("  ");
        for &(gh, bh) in &heights {
            line.push(if gh >= level { '▓' } else { ' ' });
            line.push(if bh >= level { '░' } else { ' ' });
        }
        let _ = writeln!(out, "{}", line.trim_end());
    }
    let mut axis = String::from("  └");
    axis.push_str(&"─".repeat(width));
    let _ = writeln!(out, "{axis}");
    let _ = writeln!(
        out,
        "   {min:.3}{}{max:.3}",
        " ".repeat(width.saturating_sub(12))
    );
    out
}

fn human_duration(ms: u64) -> String {
    if ms < 1000 {
        format!("{ms} ms")
    } else {
        format!("{:.1} s", ms as f64 / 1000.0)
    }
}

fn ellipsis_tail(s: &str, max: usize) -> String {
    let len = s.chars().count();
    if len <= max {
        return s.to_string();
    }
    let skip = len - max + 1;
    format!("…{}", s.chars().skip(skip).collect::<String>())
}

fn escape_md(s: &str) -> String {
    s.replace('|', "\\|").replace('`', "\\`")
}

fn reproduce_command(report: &FindReport) -> String {
    let ds = &report.dataset;
    let mut cmd = format!(
        "duelens find {} --good-label {} --bad-label {} --objective {} --seed {} --top {}",
        shell_quote(&ds.root),
        shell_quote(&ds.good_label),
        shell_quote(&ds.bad_label),
        shell_quote(&report.config.objective),
        report.config.seed,
        report.config.top_k
    );
    if let Some(manifest) = &ds.manifest {
        cmd.push_str(&format!(" --manifest {}", shell_quote(manifest)));
    }
    if ds.recursive {
        cmd.push_str(" --recursive");
    }
    cmd
}

fn shell_quote(s: &str) -> String {
    if s.chars()
        .all(|c| c.is_ascii_alphanumeric() || "_./-".contains(c))
    {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', "'\\''"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_renders_two_series() {
        let scores = vec![
            ImageScore {
                path: "a".into(),
                label: "good",
                held_out: true,
                d_good: 0.0,
                d_bad: 1.0,
                score: 1.0,
                p_good: Some(0.9),
            },
            ImageScore {
                path: "b".into(),
                label: "bad",
                held_out: true,
                d_good: 1.0,
                d_bad: 0.0,
                score: -1.0,
                p_good: Some(0.1),
            },
        ];
        let h = histogram(&scores, 10, 4);
        assert!(h.contains('▓'));
        assert!(h.contains('░'));
    }

    #[test]
    fn tail_ellipsis_keeps_filename() {
        assert_eq!(ellipsis_tail("short.png", 20), "short.png");
        assert!(ellipsis_tail("/a/very/long/path/image.png", 12).ends_with("image.png"));
    }

    #[test]
    fn shell_quote_round_trip() {
        assert_eq!(shell_quote("plain/path.png"), "plain/path.png");
        assert_eq!(shell_quote("a b.png"), "'a b.png'");
        assert_eq!(shell_quote("it's.png"), "'it'\\''s.png'");
    }
}
