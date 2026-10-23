//! Dataset discovery and loading.
//!
//! Three layouts are supported, checked in this order:
//!
//! 1. an explicit CSV manifest (`--manifest`), with `path,label` columns,
//! 2. `root/<good-label>/` and `root/<bad-label>/` subdirectories,
//! 3. a flat directory where every image name starts with a label prefix,
//!    e.g. `good_003.png` / `bad-17.jpg`.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use serde::Serialize;

use crate::features;

pub const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp", "bmp"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Label {
    Good,
    Bad,
}

impl Label {
    pub fn as_str(self) -> &'static str {
        match self {
            Label::Good => "good",
            Label::Bad => "bad",
        }
    }
}

#[derive(Debug, Clone)]
pub struct DatasetSpec {
    pub root: PathBuf,
    pub good_label: String,
    pub bad_label: String,
    pub manifest: Option<PathBuf>,
    pub recursive: bool,
}

impl Default for DatasetSpec {
    fn default() -> Self {
        Self {
            root: PathBuf::from("."),
            good_label: "good".to_string(),
            bad_label: "bad".to_string(),
            manifest: None,
            recursive: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LabeledImage {
    pub path: PathBuf,
    pub label: Label,
}

#[derive(Debug, Clone, Serialize)]
pub struct Skipped {
    pub path: String,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct Discovery {
    pub images: Vec<LabeledImage>,
    pub skipped: Vec<Skipped>,
}

#[derive(Debug, Clone)]
pub struct Sample {
    pub path: PathBuf,
    pub label: Label,
    pub features: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct Dataset {
    pub root: PathBuf,
    pub samples: Vec<Sample>,
    pub good: Vec<usize>,
    pub bad: Vec<usize>,
    pub skipped: Vec<Skipped>,
}

impl Dataset {
    pub fn labels(&self) -> Vec<bool> {
        self.samples
            .iter()
            .map(|s| s.label == Label::Good)
            .collect()
    }

    pub fn n(&self) -> usize {
        self.samples.len()
    }
}

/// Find labeled images without decoding them.
pub fn discover(spec: &DatasetSpec) -> Result<Discovery> {
    if spec.good_label.eq_ignore_ascii_case(&spec.bad_label) {
        bail!(
            "--good-label and --bad-label are both {:?}; they must differ",
            spec.good_label
        );
    }
    if spec.good_label.trim().is_empty() || spec.bad_label.trim().is_empty() {
        bail!("labels must not be empty");
    }

    if let Some(manifest) = &spec.manifest {
        return discover_manifest(manifest, &spec.good_label, &spec.bad_label);
    }

    if !spec.root.is_dir() {
        bail!(
            "{} is not a directory (point --manifest at a CSV for a single-file dataset)",
            spec.root.display()
        );
    }

    let good_dir = spec.root.join(&spec.good_label);
    let bad_dir = spec.root.join(&spec.bad_label);
    let has_good = good_dir.is_dir();
    let has_bad = bad_dir.is_dir();

    if has_good || has_bad {
        if !(has_good && has_bad) {
            let missing = if has_good { bad_dir } else { good_dir };
            bail!(
                "found only one label directory; {} is missing",
                missing.display()
            );
        }
        let mut good_images = Vec::new();
        let mut bad_images = Vec::new();
        let mut skipped = Vec::new();
        collect_images(&good_dir, spec.recursive, &mut good_images, &mut skipped)?;
        collect_images(&bad_dir, spec.recursive, &mut bad_images, &mut skipped)?;
        if good_images.is_empty() || bad_images.is_empty() {
            bail!(
                "need images in both {} and {}",
                good_dir.display(),
                bad_dir.display()
            );
        }
        let mut labeled: Vec<LabeledImage> = good_images
            .into_iter()
            .map(|path| LabeledImage {
                path,
                label: Label::Good,
            })
            .chain(bad_images.into_iter().map(|path| LabeledImage {
                path,
                label: Label::Bad,
            }))
            .collect();
        labeled.sort_by(|a, b| {
            (a.label == Label::Bad)
                .cmp(&(b.label == Label::Bad))
                .then_with(|| a.path.cmp(&b.path))
        });
        return Ok(Discovery {
            images: labeled,
            skipped,
        });
    }

    // Flat layout with filename prefixes.
    let mut files = Vec::new();
    let mut skipped = Vec::new();
    collect_images(&spec.root, spec.recursive, &mut files, &mut skipped)?;
    let mut images = Vec::new();
    for path in files {
        match label_from_filename(&path, &spec.good_label, &spec.bad_label) {
            Some(label) => images.push(LabeledImage { path, label }),
            None => skipped.push(Skipped {
                path: path.display().to_string(),
                reason: format!(
                    "filename does not start with {:?} or {:?}",
                    spec.good_label, spec.bad_label
                ),
            }),
        }
    }
    if images.is_empty() {
        bail!(
            "no labeled images found under {}.\n\
             Expected {}/ and {}/ subdirectories, or filenames such as {}_{}.png / {}_{}.png.\n\
             Run `duelens demo` to generate a sample dataset.",
            spec.root.display(),
            spec.good_label,
            spec.bad_label,
            spec.good_label,
            "001",
            spec.bad_label,
            "001"
        );
    }
    images.sort_by(|a, b| {
        (a.label == Label::Bad)
            .cmp(&(b.label == Label::Bad))
            .then_with(|| a.path.cmp(&b.path))
    });
    Ok(Discovery { images, skipped })
}

fn label_from_filename(path: &Path, good: &str, bad: &str) -> Option<Label> {
    let name = path.file_name()?.to_string_lossy().to_lowercase();
    if has_label_prefix(&name, &good.to_lowercase()) {
        Some(Label::Good)
    } else if has_label_prefix(&name, &bad.to_lowercase()) {
        Some(Label::Bad)
    } else {
        None
    }
}

fn has_label_prefix(name: &str, label: &str) -> bool {
    match name.strip_prefix(label) {
        Some(rest) => rest
            .chars()
            .next()
            .is_none_or(|c| matches!(c, '_' | '-' | '.' | ' ')),
        None => false,
    }
}

fn collect_images(
    dir: &Path,
    recursive: bool,
    out: &mut Vec<PathBuf>,
    skipped: &mut Vec<Skipped>,
) -> Result<()> {
    let mut entries: Vec<fs::DirEntry> = fs::read_dir(dir)
        .with_context(|| format!("cannot read directory {}", dir.display()))?
        .collect::<std::io::Result<Vec<_>>>()
        .with_context(|| format!("cannot list directory {}", dir.display()))?;
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let path = entry.path();
        let file_type = entry
            .file_type()
            .with_context(|| format!("cannot stat {}", path.display()))?;
        if file_type.is_dir() {
            if recursive {
                let name = entry.file_name();
                if !name.to_string_lossy().starts_with('.') {
                    collect_images(&path, recursive, out, skipped)?;
                }
            }
        } else if file_type.is_file() {
            if has_image_extension(&path) {
                out.push(path);
            } else {
                skipped.push(Skipped {
                    path: path.display().to_string(),
                    reason: "not a supported image extension".to_string(),
                });
            }
        }
    }
    Ok(())
}

fn has_image_extension(path: &Path) -> bool {
    path.extension()
        .map(|e| IMAGE_EXTENSIONS.contains(&e.to_string_lossy().to_lowercase().as_str()))
        .unwrap_or(false)
}

fn discover_manifest(path: &Path, good: &str, bad: &str) -> Result<Discovery> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_path(path)
        .with_context(|| format!("cannot open manifest {}", path.display()))?;
    let base = path.parent().unwrap_or(Path::new("."));
    let mut records = reader.records();
    let mut skipped = Vec::new();
    let mut images = Vec::new();
    let mut seen = BTreeSet::new();

    let mut path_col = 0usize;
    let mut label_col = 1usize;
    let mut first = true;

    for record in records.by_ref() {
        let record = match record {
            Ok(r) => r,
            Err(e) => {
                skipped.push(Skipped {
                    path: path.display().to_string(),
                    reason: format!("malformed CSV row: {e}"),
                });
                continue;
            }
        };
        if record.iter().all(|f| f.trim().is_empty()) {
            continue;
        }
        if first {
            first = false;
            let lower: Vec<String> = record.iter().map(|f| f.trim().to_lowercase()).collect();
            let p = lower
                .iter()
                .position(|f| matches!(f.as_str(), "path" | "file" | "image" | "filename"));
            let l = lower.iter().position(|f| f == "label");
            if let (Some(p), Some(l)) = (p, l) {
                path_col = p;
                label_col = l;
                continue; // header consumed
            }
        }

        let Some(raw_path) = record.get(path_col).map(str::trim) else {
            continue;
        };
        if raw_path.is_empty() {
            continue;
        }
        let label_text = record.get(label_col).unwrap_or("").trim();
        let label = if label_text.eq_ignore_ascii_case(good) {
            Label::Good
        } else if label_text.eq_ignore_ascii_case(bad) {
            Label::Bad
        } else {
            skipped.push(Skipped {
                path: raw_path.to_string(),
                reason: format!("label {label_text:?} is neither {good:?} nor {bad:?}"),
            });
            continue;
        };
        let p = Path::new(raw_path);
        let full = if p.is_absolute() {
            p.to_path_buf()
        } else {
            base.join(p)
        };
        let key = fs::canonicalize(&full).unwrap_or_else(|_| full.clone());
        if !seen.insert(key) {
            skipped.push(Skipped {
                path: full.display().to_string(),
                reason: "duplicate path".to_string(),
            });
            continue;
        }
        images.push(LabeledImage { path: full, label });
    }

    if images.is_empty() {
        bail!(
            "manifest {} produced no usable rows with labels {good:?}/{bad:?}",
            path.display()
        );
    }
    images.sort_by(|a, b| {
        (a.label == Label::Bad)
            .cmp(&(b.label == Label::Bad))
            .then_with(|| a.path.cmp(&b.path))
    });
    Ok(Discovery { images, skipped })
}

/// Discover and decode images, extracting one descriptor per image.
///
/// Images that fail to decode are skipped and reported; at least two images per
/// label are required so that leave-two-out separation is measurable.
pub fn load(spec: &DatasetSpec) -> Result<Dataset> {
    let discovery = discover(spec)?;
    let loaded: Vec<(LabeledImage, Result<Vec<f32>>)> = discovery
        .images
        .into_par_iter()
        .map(|image| {
            let features = features::extract_path(&image.path);
            (image, features)
        })
        .collect();

    let mut samples = Vec::with_capacity(loaded.len());
    let mut skipped = discovery.skipped;
    for (image, result) in loaded {
        match result {
            Ok(features) => samples.push(Sample {
                path: image.path,
                label: image.label,
                features,
            }),
            Err(err) => skipped.push(Skipped {
                path: image.path.display().to_string(),
                reason: format!("{err:#}"),
            }),
        }
    }

    let good: Vec<usize> = samples
        .iter()
        .enumerate()
        .filter(|(_, s)| s.label == Label::Good)
        .map(|(i, _)| i)
        .collect();
    let bad: Vec<usize> = samples
        .iter()
        .enumerate()
        .filter(|(_, s)| s.label == Label::Bad)
        .map(|(i, _)| i)
        .collect();

    if good.len() < 2 || bad.len() < 2 {
        bail!(
            "need at least 2 good and 2 bad decodable images (found {} good, {} bad).\n\
             A pair is scored on held-out images, so one image per class is not enough.",
            good.len(),
            bad.len()
        );
    }

    Ok(Dataset {
        root: spec.root.clone(),
        samples,
        good,
        bad,
        skipped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;
    use tempfile::tempdir;

    fn write_png(path: &Path) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        RgbImage::from_pixel(4, 4, image::Rgb([10, 20, 30]))
            .save(path)
            .unwrap();
    }

    #[test]
    fn discovers_label_subdirectories() {
        let dir = tempdir().unwrap();
        write_png(&dir.path().join("good/a.png"));
        write_png(&dir.path().join("good/b.png"));
        write_png(&dir.path().join("bad/c.png"));
        write_png(&dir.path().join("bad/d.png"));
        let spec = DatasetSpec {
            root: dir.path().to_path_buf(),
            ..Default::default()
        };
        let found = discover(&spec).unwrap();
        assert_eq!(found.images.len(), 4);
        assert_eq!(
            found
                .images
                .iter()
                .filter(|i| i.label == Label::Good)
                .count(),
            2
        );
    }

    #[test]
    fn discovers_flat_prefixes() {
        let dir = tempdir().unwrap();
        write_png(&dir.path().join("good_001.png"));
        write_png(&dir.path().join("good-002.png"));
        write_png(&dir.path().join("bad.1.png"));
        write_png(&dir.path().join("bad 2.png"));
        write_png(&dir.path().join("unrelated.png"));
        let spec = DatasetSpec {
            root: dir.path().to_path_buf(),
            ..Default::default()
        };
        let found = discover(&spec).unwrap();
        assert_eq!(found.images.len(), 4);
        assert_eq!(found.skipped.len(), 1);
    }

    #[test]
    fn discovers_manifest_with_header() {
        let dir = tempdir().unwrap();
        write_png(&dir.path().join("imgs/a.png"));
        write_png(&dir.path().join("imgs/b.png"));
        let manifest = dir.path().join("labels.csv");
        fs::write(&manifest, "path,label\nimgs/a.png,good\nimgs/b.png,bad\n").unwrap();
        let spec = DatasetSpec {
            root: dir.path().to_path_buf(),
            manifest: Some(manifest),
            ..Default::default()
        };
        let found = discover(&spec).unwrap();
        assert_eq!(found.images.len(), 2);
        assert!(found.images[0].path.ends_with("imgs/a.png"));
        assert_eq!(found.images[0].label, Label::Good);
    }

    #[test]
    fn custom_label_names() {
        let dir = tempdir().unwrap();
        write_png(&dir.path().join("keep/1.png"));
        write_png(&dir.path().join("reject/1.png"));
        let spec = DatasetSpec {
            root: dir.path().to_path_buf(),
            good_label: "keep".into(),
            bad_label: "reject".into(),
            ..Default::default()
        };
        let found = discover(&spec).unwrap();
        assert_eq!(found.images.len(), 2);
        assert_eq!(found.images[0].label, Label::Good);
        assert_eq!(found.images[1].label, Label::Bad);
    }
}
