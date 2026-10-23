//! Synthetic demo dataset: circles (good) vs squares/rectangles (bad).
//!
//! Colour and background are randomized independently of the label, so the only
//! reliable cue is shape. A few images per class are crisp and centred ("clean")
//! while the rest are blurred, low-contrast and noisy ("rough"); a good exemplar
//! search should surface the clean prototypes.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use image::{Rgb, RgbImage, imageops};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

pub const DEFAULT_DEMO_DIR: &str = ".duelens-demo";

#[derive(Debug, Clone)]
pub struct DemoSpec {
    pub out: PathBuf,
    pub per_class: usize,
    pub clean_per_class: usize,
    pub size: u32,
    pub seed: u64,
    pub force: bool,
}

impl Default for DemoSpec {
    fn default() -> Self {
        Self {
            out: PathBuf::from(DEFAULT_DEMO_DIR),
            per_class: 18,
            clean_per_class: 5,
            size: 96,
            seed: 7,
            force: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DemoData {
    pub out: PathBuf,
    pub good: Vec<PathBuf>,
    pub bad: Vec<PathBuf>,
}

/// Write a fresh demo dataset. Refuses to touch an existing directory unless
/// `force` is set; with `force` the directory is replaced.
pub fn generate(spec: &DemoSpec) -> Result<DemoData> {
    if spec.per_class < 2 {
        bail!("--per-class must be at least 2");
    }
    if spec.clean_per_class > spec.per_class {
        bail!("--clean-per-class cannot exceed --per-class");
    }
    if spec.size < 32 {
        bail!("--size must be at least 32");
    }

    if spec.out.exists() {
        if !spec.force {
            bail!(
                "{} already exists; pass --force to replace it",
                spec.out.display()
            );
        }
        let absolute = fs::canonicalize(&spec.out)?;
        if !absolute.is_dir() {
            bail!("{} exists and is not a directory", absolute.display());
        }
        let cwd = fs::canonicalize(std::env::current_dir()?)?;
        let home = std::env::var_os("HOME").map(PathBuf::from);
        if absolute == Path::new("/")
            || absolute == cwd
            || home.as_ref().is_some_and(|h| absolute == *h)
        {
            bail!("refusing to remove {}", absolute.display());
        }
        fs::remove_dir_all(&absolute)
            .with_context(|| format!("cannot replace {}", absolute.display()))?;
    }

    let mut rng = SmallRng::seed_from_u64(spec.seed);
    let good_dir = spec.out.join("good");
    let bad_dir = spec.out.join("bad");
    fs::create_dir_all(&good_dir)?;
    fs::create_dir_all(&bad_dir)?;

    let mut good = Vec::with_capacity(spec.per_class);
    let mut bad = Vec::with_capacity(spec.per_class);
    for (label, dir, circle, out) in [
        ("good", &good_dir, true, &mut good),
        ("bad", &bad_dir, false, &mut bad),
    ] {
        for i in 0..spec.per_class {
            let clean = i < spec.clean_per_class;
            let image = render(&mut rng, spec.size, circle, clean);
            let name = format!(
                "{label}_{i:03}_{}.png",
                if clean { "clean" } else { "rough" }
            );
            let path = dir.join(name);
            image
                .save(&path)
                .with_context(|| format!("cannot save {}", path.display()))?;
            out.push(path);
        }
    }

    Ok(DemoData {
        out: spec.out.clone(),
        good,
        bad,
    })
}

fn render(rng: &mut SmallRng, size: u32, circle: bool, clean: bool) -> RgbImage {
    let mut img = RgbImage::new(size, size);
    let size_f = size as f32;

    // Random background gradient: hue and direction are label-independent.
    // Clean prototypes get a quiet, low-saturation backdrop so that shape is the
    // dominant cue; rough images get a busier, more colourful one.
    let bg_hue = rng.random_range(0.0..1.0);
    let (bg_sat, bg_val, hue_jitter, val_jitter) = if clean {
        (
            rng.random_range(0.0..0.12),
            rng.random_range(0.56..0.78),
            0.02,
            0.04,
        )
    } else {
        (
            rng.random_range(0.12..0.55),
            rng.random_range(0.30..0.85),
            0.12,
            0.25,
        )
    };
    let c1 = hsv(bg_hue, bg_sat, bg_val);
    let c2 = hsv(
        (bg_hue + rng.random_range(-hue_jitter..hue_jitter)).rem_euclid(1.0),
        (bg_sat + rng.random_range(-0.1..0.1)).clamp(0.0, 0.8),
        (bg_val + rng.random_range(-val_jitter..val_jitter)).clamp(0.1, 0.95),
    );
    let theta = rng.random_range(0.0..std::f32::consts::TAU);
    let (sin, cos) = theta.sin_cos();

    // Shape colour: force a luma gap against the background so both classes are
    // equally visible.
    let bg_luma = 0.299 * c1[0] as f32 + 0.587 * c1[1] as f32 + 0.114 * c1[2] as f32;
    let shape_val = if bg_luma > 127.0 {
        rng.random_range(0.04..0.32)
    } else {
        rng.random_range(0.68..0.98)
    };
    let shape = hsv(
        rng.random_range(0.0..1.0),
        rng.random_range(0.55..1.0),
        shape_val,
    );

    let noise = if clean { 3.0 } else { 20.0 };
    for y in 0..size {
        for x in 0..size {
            let t = 0.5
                + 0.5 * ((x as f32 - size_f / 2.0) * cos + (y as f32 - size_f / 2.0) * sin)
                    / (size_f * std::f32::consts::FRAC_1_SQRT_2);
            let mut p = [0f32; 3];
            for c in 0..3 {
                let base = c1[c] as f32 * (1.0 - t) + c2[c] as f32 * t;
                p[c] = base + rng.random_range(-noise..noise);
            }
            img.put_pixel(x, y, Rgb(clamp_rgb(p)));
        }
    }

    // Shape placement and size.
    let jitter = if clean { 0.05 } else { 0.12 };
    let cx = size_f / 2.0 + rng.random_range(-jitter..jitter) * size_f;
    let cy = size_f / 2.0 + rng.random_range(-jitter..jitter) * size_f;
    let half = if circle {
        let radius = rng.random_range(0.21..0.31) * size_f * if clean { 1.0 } else { 0.85 };
        (radius, radius)
    } else {
        let hw = rng.random_range(0.17..0.28) * size_f * if clean { 1.0 } else { 0.9 };
        let hh = rng.random_range(0.17..0.28) * size_f * if clean { 1.0 } else { 0.9 };
        (hw, hh)
    };

    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 + 0.5 - cx;
            let dy = y as f32 + 0.5 - cy;
            let coverage = if circle {
                let dist = (dx * dx + dy * dy).sqrt();
                ((half.0 - dist) / 1.4).clamp(0.0, 1.0)
            } else {
                let fx = ((half.0 - dx.abs()) / 1.4).clamp(0.0, 1.0);
                let fy = ((half.1 - dy.abs()) / 1.4).clamp(0.0, 1.0);
                fx * fy
            };
            if coverage <= 0.0 {
                continue;
            }
            let mut px = *img.get_pixel(x, y);
            for c in 0..3 {
                let v = px[c] as f32 * (1.0 - coverage) + shape[c] as f32 * coverage;
                px[c] = v.round().clamp(0.0, 255.0) as u8;
            }
            let px = Rgb(clamp_rgb([px[0] as f32, px[1] as f32, px[2] as f32]));
            img.put_pixel(x, y, px);
        }
    }

    if !clean {
        img = imageops::blur(&img, rng.random_range(0.7..1.8));
        let mean = mean_luma(&img);
        let contrast = rng.random_range(0.45..0.75);
        // A random colour cast makes rough shots poor references, not just soft
        // ones: their statistics drift away from the class centre.
        let gains = [
            rng.random_range(0.72..1.28),
            rng.random_range(0.72..1.28),
            rng.random_range(0.72..1.28),
        ];
        for pixel in img.pixels_mut() {
            for c in 0..3 {
                let v = pixel[c] as f32;
                let out = (mean + (v - mean) * contrast) * gains[c] + rng.random_range(-10.0..10.0);
                pixel[c] = out.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    img
}

fn mean_luma(img: &RgbImage) -> f32 {
    let mut sum = 0f64;
    for p in img.pixels() {
        sum += 0.299 * p[0] as f64 + 0.587 * p[1] as f64 + 0.114 * p[2] as f64;
    }
    (sum / img.pixels().len() as f64) as f32
}

fn clamp_rgb(v: [f32; 3]) -> [u8; 3] {
    [
        v[0].round().clamp(0.0, 255.0) as u8,
        v[1].round().clamp(0.0, 255.0) as u8,
        v[2].round().clamp(0.0, 255.0) as u8,
    ]
}

fn hsv(h: f32, s: f32, v: f32) -> [u8; 3] {
    let h = h.rem_euclid(1.0) * 6.0;
    let i = h.floor() as i32;
    let f = h - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    let (r, g, b) = match i.rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    clamp_rgb([r * 255.0, g * 255.0, b * 255.0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn writes_expected_files() {
        let dir = tempdir().unwrap();
        let out = dir.path().join("demo");
        let data = generate(&DemoSpec {
            out: out.clone(),
            per_class: 6,
            clean_per_class: 2,
            size: 48,
            seed: 1,
            force: false,
        })
        .unwrap();
        assert_eq!(data.good.len(), 6);
        assert_eq!(data.bad.len(), 6);
        assert!(data.good[0].ends_with("good_000_clean.png"));
        assert!(data.good[5].ends_with("good_005_rough.png"));
        for path in data.good.iter().chain(&data.bad) {
            let img = image::open(path).unwrap();
            assert_eq!(img.width(), 48);
        }
    }

    #[test]
    fn refuses_to_overwrite_without_force() {
        let dir = tempdir().unwrap();
        let out = dir.path().join("demo");
        fs::create_dir_all(&out).unwrap();
        fs::write(out.join("marker"), "keep").unwrap();
        let err = generate(&DemoSpec {
            out: out.clone(),
            ..Default::default()
        })
        .unwrap_err();
        assert!(err.to_string().contains("--force"));
        assert!(out.join("marker").exists());
    }
}
