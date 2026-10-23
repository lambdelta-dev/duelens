use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};

use duelens::report as output;
use duelens::search::{FindOptions, Objective, run_find};
use duelens::synth::{self, DemoSpec};

#[derive(Parser)]
#[command(
    name = "duelens",
    version,
    about = "Find the good/bad image pair that draws the sharpest decision boundary",
    long_about = "duelens scores every (good example, bad example) pair in a labeled image set \
                  and returns the pair that best separates the two classes when used as reference \
                  images for a few-shot decision API.\n\n\
                  Input layouts:\n\
                  1. root/good/ and root/bad/ subdirectories\n\
                  2. a CSV manifest with path,label columns (--manifest)\n\
                  3. a flat directory with good_*.png / bad_*.png style names\n\n\
                  The default scorer is a classical-CV descriptor, a fast stand-in for the real \
                  decision API. Pair selection is exhaustive up to --max-exhaustive, then beam search."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Search a labeled image set for the best exemplar pair.
    Find(FindArgs),
    /// Generate a synthetic circles-vs-rectangles dataset and pick its best pair.
    Demo(DemoArgs),
}

#[derive(Args)]
struct FindArgs {
    /// Directory containing the labeled images.
    #[arg(default_value = ".")]
    path: PathBuf,

    /// Label of the "good" class (directory name or filename prefix).
    #[arg(long, default_value = "good")]
    good_label: String,

    /// Label of the "bad" class (directory name or filename prefix).
    #[arg(long, default_value = "bad")]
    bad_label: String,

    /// CSV file with path,label columns; overrides directory discovery.
    #[arg(long)]
    manifest: Option<PathBuf>,

    /// Recurse into subdirectories below the label directories.
    #[arg(long)]
    recursive: bool,

    /// Metric used to rank pairs.
    #[arg(long, value_enum, default_value = "auc")]
    objective: Objective,

    /// How many ranked pairs to report.
    #[arg(long, default_value_t = 10)]
    top: usize,

    /// Random seed for sampling, beam search and the bootstrap CI.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Exhaustively evaluate up to this many pairs, then switch to beam search.
    #[arg(long, default_value_t = 200_000)]
    max_exhaustive: usize,

    /// Beam width for large candidate spaces.
    #[arg(long, default_value_t = 16)]
    beam_width: usize,

    /// Skip the bootstrap AUC confidence interval.
    #[arg(long)]
    no_bootstrap: bool,

    /// Write the full result as JSON.
    #[arg(long, value_name = "FILE")]
    json: Option<PathBuf>,

    /// Write a Markdown report.
    #[arg(long, value_name = "FILE")]
    report: Option<PathBuf>,

    /// Copy the winning pair into this directory as good.*/bad.* plus pair.json.
    #[arg(long, value_name = "DIR")]
    export: Option<PathBuf>,

    /// Suppress progress messages on stderr.
    #[arg(long)]
    quiet: bool,
}

#[derive(Args)]
struct DemoArgs {
    /// Directory to generate the dataset in (replaced with --force).
    #[arg(long, default_value = synth::DEFAULT_DEMO_DIR)]
    out: PathBuf,

    /// Images per class.
    #[arg(long, default_value_t = 18)]
    per_class: usize,

    /// How many of each class are crisp, centred prototypes.
    #[arg(long, default_value_t = 5)]
    clean_per_class: usize,

    /// Image side length in pixels.
    #[arg(long, default_value_t = 96)]
    size: u32,

    /// Dataset and search seed.
    #[arg(long, default_value_t = 7)]
    seed: u64,

    /// Replace an existing output directory.
    #[arg(long)]
    force: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Find(args) => run_find_command(args),
        Command::Demo(args) => run_demo_command(args),
    }
}

fn run_find_command(args: FindArgs) -> Result<()> {
    let options = FindOptions {
        root: args.path,
        good_label: args.good_label,
        bad_label: args.bad_label,
        manifest: args.manifest,
        recursive: args.recursive,
        objective: args.objective,
        top_k: args.top,
        seed: args.seed,
        max_exhaustive: args.max_exhaustive,
        beam_width: args.beam_width,
        bootstrap_resamples: if args.no_bootstrap { 0 } else { 400 },
        verbose: !args.quiet,
    };
    let report = run_find(&options)?;
    print!("{}", output::render_terminal(&report));

    if let Some(path) = args.json {
        output::write_json(&report, &path)?;
        println!("\nwrote {}", path.display());
    }
    if let Some(path) = args.report {
        output::write_markdown(&report, &path)?;
        println!("wrote {}", path.display());
    }
    if let Some(dir) = args.export {
        let files = output::export_winner(&report, &dir)?;
        for file in files {
            println!("wrote {}", file.display());
        }
    }
    Ok(())
}

fn run_demo_command(args: DemoArgs) -> Result<()> {
    println!(
        "duelens demo: generating {} good (circles) + {} bad (rectangles) images ({}px, seed {})",
        args.per_class, args.per_class, args.size, args.seed
    );
    let data = synth::generate(&DemoSpec {
        out: args.out,
        per_class: args.per_class,
        clean_per_class: args.clean_per_class,
        size: args.size,
        seed: args.seed,
        force: args.force,
    })?;
    println!(
        "wrote {} images to {} ({} clean prototypes per class)",
        data.good.len() + data.bad.len(),
        data.out.display(),
        args.clean_per_class
    );

    let report = run_find(&FindOptions {
        root: data.out.clone(),
        seed: args.seed,
        top_k: 8,
        ..Default::default()
    })?;
    print!("{}", output::render_terminal(&report));

    let report_path = data.out.join("report.md");
    let picked_dir = data.out.join("picked");
    output::write_markdown(&report, &report_path)?;
    output::export_winner(&report, &picked_dir)?;
    println!("\nwrote {}", report_path.display());
    println!("wrote {}/", picked_dir.display());

    if let Some(winner) = &report.winner {
        let picked_clean = ["good", "bad"]
            .iter()
            .filter(|role| {
                let path = if **role == "good" {
                    &winner.good
                } else {
                    &winner.bad
                };
                path.contains("_clean")
            })
            .count();
        println!(
            "winner: AUC {:.3}, balanced accuracy {:.3} ({} of 2 reference images are crisp prototypes; \
             the search optimizes separation, not prettiness)",
            winner.auc, winner.balanced_accuracy, picked_clean
        );
    }
    println!(
        "\nNext: duelens find {} --objective blend --json {}",
        data.out.display(),
        data.out.join("result.json").display()
    );
    Ok(())
}
