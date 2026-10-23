//! Classical-CV image descriptors.
//!
//! The default surrogate for a decision API is a compact, deterministic visual
//! descriptor built from cheap, interpretable pieces:
//!
//! * a 16x16 luma thumbnail (layout / shape),
//! * a 4x4x4 RGB histogram (palette),
//! * a HOG-lite gradient histogram on a 4x4 grid (structure / edges),
//! * a 4x4 spatial colour grid (where the colours are),
//! * seven global statistics (brightness, contrast, saturation, colourfulness,
//!   edge density, orientation entropy, centre-surround contrast).
//!
//! Every block is L2-normalised, weighted and concatenated; the final vector is
//! L2-normalised again. Images are centre-cropped to a square and resized to
//! 64x64 first, so the descriptor is resolution- and aspect-independent.
//!
//! This is deliberately *not* a semantic embedding. It is a fast, dependency
//! free proxy that makes the selection search runnable on any machine. Replace
//! it with CLIP/DINO/etc. vectors for production use; the search only sees the
//! distance matrix.

use std::path::Path;

use anyhow::{Context, Result};
use image::{RgbImage, imageops};

/// Bumped whenever the descriptor layout or weights change.
pub const FEATURE_VERSION: &str = "classical-v1";

/// Working resolution after centre-cropping.
pub const ANALYSIS: u32 = 64;

const THUMB: usize = 16;
const COLOR_BINS: usize = 4;
const HOG_CELLS: usize = 4;
const HOG_BINS: usize = 8;
const SPATIAL_CELLS: usize = 4;
const STATS: usize = 7;

const W_THUMB: f32 = 1.0;
const W_COLOR: f32 = 0.9;
const W_HOG: f32 = 1.2;
const W_SPATIAL: f32 = 0.9;
const W_STATS: f32 = 0.7;

/// Dimension of [`extract_rgb`]'s output.
pub const DIM: usize = THUMB * THUMB
    + COLOR_BINS * COLOR_BINS * COLOR_BINS
    + HOG_CELLS * HOG_CELLS * HOG_BINS
    + SPATIAL_CELLS * SPATIAL_CELLS * 3
    + STATS;

/// Decode an image and compute its descriptor.
pub fn extract_path(path: &Path) -> Result<Vec<f32>> {
    let reader = image::ImageReader::open(path)
        .with_context(|| format!("cannot open {}", path.display()))?
        .with_guessed_format()
        .with_context(|| format!("cannot sniff image format for {}", path.display()))?;
    let decoded = reader
        .decode()
        .with_context(|| format!("cannot decode {}", path.display()))?;
    Ok(extract_rgb(&decoded.to_rgb8()))
}

/// Compute the descriptor for an already-decoded RGB image.
pub fn extract_rgb(img: &RgbImage) -> Vec<f32> {
    let small = centre_crop_square(img);
    let n = (ANALYSIS * ANALYSIS) as usize;

    let mut rgb = vec![[0f32; 3]; n];
    let mut luma = vec![0f32; n];
    for (i, pixel) in small.pixels().enumerate() {
        let [r, g, b] = pixel.0;
        let rf = r as f32 / 255.0;
        let gf = g as f32 / 255.0;
        let bf = b as f32 / 255.0;
        rgb[i] = [rf, gf, bf];
        luma[i] = 0.299 * rf + 0.587 * gf + 0.114 * bf;
    }

    let mut out = Vec::with_capacity(DIM);

    // 1. Luma thumbnail: average 4x4 blocks down to 16x16.
    let mut thumb = vec![0f32; THUMB * THUMB];
    let factor = ANALYSIS as usize / THUMB;
    for ty in 0..THUMB {
        for tx in 0..THUMB {
            let mut sum = 0f32;
            for dy in 0..factor {
                let y = ty * factor + dy;
                for dx in 0..factor {
                    let x = tx * factor + dx;
                    sum += luma[y * ANALYSIS as usize + x];
                }
            }
            thumb[ty * THUMB + tx] = sum / (factor * factor) as f32;
        }
    }
    push_block(&mut out, &mut thumb, W_THUMB);

    // 2. RGB histogram, Hellinger-transformed (sqrt of frequencies).
    let mut hist = vec![0f32; COLOR_BINS * COLOR_BINS * COLOR_BINS];
    for [r, g, b] in &rgb {
        let qr = ((r * COLOR_BINS as f32) as usize).min(COLOR_BINS - 1);
        let qg = ((g * COLOR_BINS as f32) as usize).min(COLOR_BINS - 1);
        let qb = ((b * COLOR_BINS as f32) as usize).min(COLOR_BINS - 1);
        hist[(qr * COLOR_BINS + qg) * COLOR_BINS + qb] += 1.0;
    }
    for v in &mut hist {
        *v = (*v / n as f32).sqrt();
    }
    push_block(&mut out, &mut hist, W_COLOR);

    // 3. HOG-lite: 8 unsigned orientation bins over 4x4 cells.
    let mut hog = vec![0f32; HOG_CELLS * HOG_CELLS * HOG_BINS];
    let side = ANALYSIS as usize;
    let mut global_orient = [0f32; HOG_BINS];
    let mut grad_mass = 0f64;
    for y in 1..side - 1 {
        for x in 1..side - 1 {
            let l = |yy: usize, xx: usize| luma[yy * side + xx];
            let gx = (l(y - 1, x + 1) + 2.0 * l(y, x + 1) + l(y + 1, x + 1))
                - (l(y - 1, x - 1) + 2.0 * l(y, x - 1) + l(y + 1, x - 1));
            let gy = (l(y + 1, x - 1) + 2.0 * l(y + 1, x) + l(y + 1, x + 1))
                - (l(y - 1, x - 1) + 2.0 * l(y - 1, x) + l(y - 1, x + 1));
            let mag = (gx * gx + gy * gy).sqrt();
            grad_mass += mag as f64;
            let mut angle = gy.atan2(gx);
            if angle < 0.0 {
                angle += std::f32::consts::PI;
            }
            if angle >= std::f32::consts::PI {
                angle -= std::f32::consts::PI;
            }
            let bin = ((angle / std::f32::consts::PI) * HOG_BINS as f32)
                .floor()
                .min((HOG_BINS - 1) as f32) as usize;
            let cx = x * HOG_CELLS / side;
            let cy = y * HOG_CELLS / side;
            hog[(cy * HOG_CELLS + cx) * HOG_BINS + bin] += mag;
            global_orient[bin] += mag;
        }
    }
    for cell in 0..HOG_CELLS * HOG_CELLS {
        let start = cell * HOG_BINS;
        let sum: f32 = hog[start..start + HOG_BINS].iter().sum();
        if sum > 0.0 {
            for v in &mut hog[start..start + HOG_BINS] {
                *v /= sum;
            }
        }
    }
    push_block(&mut out, &mut hog, W_HOG);

    // 4. Spatial colour: mean RGB in each 4x4 cell.
    let mut spatial = vec![0f32; SPATIAL_CELLS * SPATIAL_CELLS * 3];
    let cell = side / SPATIAL_CELLS;
    for cy in 0..SPATIAL_CELLS {
        for cx in 0..SPATIAL_CELLS {
            let mut acc = [0f32; 3];
            for dy in 0..cell {
                for dx in 0..cell {
                    let idx = (cy * cell + dy) * side + (cx * cell + dx);
                    for c in 0..3 {
                        acc[c] += rgb[idx][c];
                    }
                }
            }
            let count = (cell * cell) as f32;
            let base = (cy * SPATIAL_CELLS + cx) * 3;
            for c in 0..3 {
                spatial[base + c] = acc[c] / count;
            }
        }
    }
    push_block(&mut out, &mut spatial, W_SPATIAL);

    // 5. Global statistics, all roughly in [0, 1].
    let mean_luma = (luma.iter().sum::<f32>() / n as f32).clamp(0.0, 1.0);
    let var_luma = (luma.iter().map(|v| (v - mean_luma).powi(2)).sum::<f32>() / n as f32)
        .max(0.0)
        .sqrt()
        .clamp(0.0, 1.0);

    let mut sat_sum = 0f32;
    let mut rg_vals = Vec::with_capacity(n);
    let mut yb_vals = Vec::with_capacity(n);
    for &[r, g, b] in &rgb {
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        sat_sum += if max > 1e-6 { (max - min) / max } else { 0.0 };
        rg_vals.push(r - g);
        yb_vals.push(0.5 * (r + g) - b);
    }
    let mean_sat = (sat_sum / n as f32).clamp(0.0, 1.0);
    let colorfulness = {
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let std = |v: &[f32], m: f32| {
            (v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt()
        };
        let rg_m = mean(&rg_vals);
        let yb_m = mean(&yb_vals);
        let rg_s = std(&rg_vals, rg_m);
        let yb_s = std(&yb_vals, yb_m);
        ((rg_s * rg_s + yb_s * yb_s).sqrt() + 0.3 * (rg_m * rg_m + yb_m * yb_m).sqrt())
            .clamp(0.0, 1.0)
    };
    let edge_density = {
        let pixels = ((side - 2) * (side - 2)) as f64;
        let mean = grad_mass / pixels;
        (mean * 4.0).min(1.0) as f32
    };
    let orientation_entropy = {
        let total: f32 = global_orient.iter().sum();
        if total <= 0.0 {
            0.0
        } else {
            let mut h = 0f32;
            for v in &global_orient {
                let p = v / total;
                if p > 0.0 {
                    h -= p * p.log2();
                }
            }
            (h / (HOG_BINS as f32).log2()).clamp(0.0, 1.0)
        }
    };
    let centre_surround = {
        let mut centre = 0f32;
        let mut centre_n = 0usize;
        let mut border = 0f32;
        let mut border_n = 0usize;
        for y in 0..side {
            for x in 0..side {
                let v = luma[y * side + x];
                let in_centre =
                    (y >= side / 4 && y < 3 * side / 4) && (x >= side / 4 && x < 3 * side / 4);
                if in_centre {
                    centre += v;
                    centre_n += 1;
                } else {
                    border += v;
                    border_n += 1;
                }
            }
        }
        let c = centre / centre_n as f32;
        let b = border / border_n as f32;
        (c - b).abs().clamp(0.0, 1.0)
    };

    let mut stats = vec![
        mean_luma,
        var_luma,
        mean_sat,
        colorfulness,
        edge_density,
        orientation_entropy,
        centre_surround,
    ];
    push_block(&mut out, &mut stats, W_STATS);

    debug_assert_eq!(out.len(), DIM);
    l2_normalise(&mut out[..]);
    out
}

fn centre_crop_square(img: &RgbImage) -> RgbImage {
    let (w, h) = img.dimensions();
    let side = w.min(h).max(1);
    let x = (w - side) / 2;
    let y = (h - side) / 2;
    let cropped = imageops::crop_imm(img, x, y, side, side).to_image();
    if side == ANALYSIS {
        cropped
    } else {
        imageops::resize(&cropped, ANALYSIS, ANALYSIS, imageops::FilterType::Triangle)
    }
}

fn push_block(out: &mut Vec<f32>, block: &mut [f32], weight: f32) {
    if l2_normalise(block) {
        out.extend(block.iter().map(|v| v * weight));
    } else {
        // Degenerate block (e.g. completely flat gradients): keep the offsets
        // stable so dimensions still line up.
        out.extend(std::iter::repeat_n(0.0, block.len()));
    }
}

/// Normalise in place; returns false for an all-zero block.
fn l2_normalise(v: &mut [f32]) -> bool {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-8 {
        return false;
    }
    for x in v {
        *x /= norm;
    }
    true
}

/// Squared Euclidean distance between two descriptors.
pub fn sq_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    fn circle_image(colour: Rgb<u8>, size: u32) -> RgbImage {
        let mut img = RgbImage::from_pixel(size, size, Rgb([0, 0, 0]));
        let r = size as f32 * 0.3;
        let c = size as f32 / 2.0;
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - c;
                let dy = y as f32 - c;
                if dx * dx + dy * dy <= r * r {
                    img.put_pixel(x, y, colour);
                }
            }
        }
        img
    }

    fn square_image(colour: Rgb<u8>, size: u32) -> RgbImage {
        let mut img = RgbImage::from_pixel(size, size, Rgb([0, 0, 0]));
        let lo = size / 4;
        let hi = 3 * size / 4;
        for y in lo..hi {
            for x in lo..hi {
                img.put_pixel(x, y, colour);
            }
        }
        img
    }

    #[test]
    fn descriptor_is_unit_length_and_finite() {
        let img = circle_image(Rgb([220, 90, 30]), 80);
        let v = extract_rgb(&img);
        assert_eq!(v.len(), DIM);
        assert!(v.iter().all(|x| x.is_finite()));
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "norm = {norm}");
    }

    #[test]
    fn descriptor_is_resolution_invariant() {
        let a = extract_rgb(&circle_image(Rgb([220, 90, 30]), 64));
        let b = extract_rgb(&circle_image(Rgb([220, 90, 30]), 128));
        let d = sq_distance(&a, &b).sqrt();
        assert!(d < 0.35, "distance across resolutions = {d}");
    }

    #[test]
    fn shape_matters_more_than_colour_change() {
        let c1 = extract_rgb(&circle_image(Rgb([220, 90, 30]), 80));
        let c2 = extract_rgb(&circle_image(Rgb([30, 160, 220]), 80));
        let s1 = extract_rgb(&square_image(Rgb([220, 90, 30]), 80));
        let same_shape = sq_distance(&c1, &c2).sqrt();
        let diff_shape = sq_distance(&c1, &s1).sqrt();
        assert!(
            diff_shape > same_shape,
            "shape distance {diff_shape} should exceed colour distance {same_shape}"
        );
    }

    #[test]
    fn flat_image_is_finite() {
        let img = RgbImage::from_pixel(64, 64, Rgb([128, 128, 128]));
        let v = extract_rgb(&img);
        assert!(v.iter().all(|x| x.is_finite()));
    }
}
