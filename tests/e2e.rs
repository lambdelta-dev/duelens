use std::fs;
use std::process::Command;

use duelens::report;
use duelens::search::{FindOptions, run_find};
use duelens::synth::{DemoSpec, generate};
use tempfile::tempdir;

fn make_demo(dir: &std::path::Path) -> std::path::PathBuf {
    let out = dir.join("demo");
    generate(&DemoSpec {
        out: out.clone(),
        per_class: 9,
        clean_per_class: 3,
        size: 64,
        seed: 11,
        force: false,
    })
    .unwrap();
    out
}

#[test]
fn library_pipeline_scores_exports_and_reports() {
    let dir = tempdir().unwrap();
    let out = make_demo(dir.path());
    let options = FindOptions {
        root: out.clone(),
        top_k: 5,
        bootstrap_resamples: 120,
        verbose: false,
        ..Default::default()
    };
    let result = run_find(&options).unwrap();

    assert_eq!(result.dataset.n_good, 9);
    assert_eq!(result.dataset.n_bad, 9);
    assert_eq!(result.top.len(), 5);
    assert_eq!(result.winner_scores.len(), 18);
    let held_out = result.winner_scores.iter().filter(|s| s.held_out).count();
    assert_eq!(held_out, 16);

    let winner = result.winner.clone().expect("winner");
    assert!(winner.auc > 0.8, "auc = {}", winner.auc);
    assert!(winner.auc <= 1.0);
    assert!(winner.balanced_accuracy > 0.7);
    assert_eq!(
        result.search.auc_best,
        result
            .top
            .iter()
            .map(|p| p.auc)
            .fold(f64::NEG_INFINITY, f64::max)
    );
    assert!(result.search.auc_worst <= result.search.auc_median);
    assert!(result.search.auc_median <= result.search.auc_best);
    assert!(result.calibration.is_some());

    let json_path = dir.path().join("out/result.json");
    report::write_json(&result, &json_path).unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&fs::read(&json_path).unwrap()).unwrap();
    assert_eq!(parsed["tool"], "duelens");
    assert!(parsed["winner"]["good"].as_str().unwrap().contains("good"));

    let md_path = dir.path().join("out/report.md");
    report::write_markdown(&result, &md_path).unwrap();
    let md = fs::read_to_string(&md_path).unwrap();
    assert!(md.contains("# duelens report"));
    assert!(md.contains("## Winner"));
    assert!(md.contains("## Score distribution"));

    let export_dir = dir.path().join("out/picked");
    let files = report::export_winner(&result, &export_dir).unwrap();
    assert_eq!(files.len(), 3);
    assert!(export_dir.join("pair.json").exists());
    let good_ext = std::path::Path::new(&winner.good)
        .extension()
        .unwrap()
        .to_string_lossy()
        .to_string();
    assert!(export_dir.join(format!("good.{good_ext}")).exists());
}

#[test]
fn cli_writes_machine_readable_output() {
    let dir = tempdir().unwrap();
    let out = make_demo(dir.path());
    let json = dir.path().join("cli/result.json");
    let bin = env!("CARGO_BIN_EXE_duelens");
    let output = Command::new(bin)
        .arg("find")
        .arg(&out)
        .arg("--quiet")
        .arg("--top")
        .arg("3")
        .arg("--json")
        .arg(&json)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("WINNER"), "stdout: {stdout}");

    let parsed: serde_json::Value = serde_json::from_slice(&fs::read(&json).unwrap()).unwrap();
    assert_eq!(parsed["tool"], "duelens");
    assert_eq!(parsed["top"].as_array().unwrap().len(), 3);
    assert_eq!(parsed["dataset"]["n_good"], 9);
    assert!(parsed["search"]["pairs_evaluated"].as_u64().unwrap() > 0);
}

#[test]
fn demo_command_generates_and_exports() {
    let dir = tempdir().unwrap();
    let out = dir.path().join("demo-cli");
    let bin = env!("CARGO_BIN_EXE_duelens");
    let output = Command::new(bin)
        .arg("demo")
        .arg("--out")
        .arg(&out)
        .arg("--per-class")
        .arg("6")
        .arg("--clean-per-class")
        .arg("2")
        .arg("--size")
        .arg("48")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(out.join("good").is_dir());
    assert!(out.join("bad").is_dir());
    assert!(out.join("report.md").exists());
    assert!(out.join("picked").join("good.png").exists());
    assert!(out.join("picked").join("bad.png").exists());
}
