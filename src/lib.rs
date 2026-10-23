//! duelens — pick the good/bad exemplar pair that draws the sharpest line.
//!
//! Given a set of labeled images, `duelens` scores every feasible
//! (good example, bad example) pair by how well that pair separates the two
//! labeled classes once used as reference images. The pair that maximises the
//! separation is the one to hand to a few-shot decision API.
//!
//! The crate is split into small pieces so the search can be reused with a real
//! API-backed scorer later:
//!
//! * [`dataset`] — discovery of the three supported input layouts,
//! * [`features`] — the default classical-CV image descriptor,
//! * [`metrics`] — AUC, balanced accuracy, effect size, calibration,
//! * [`search`] — distance matrix + exhaustive / beam exemplar search,
//! * [`report`] — terminal, JSON and Markdown output,
//! * [`synth`] — the synthetic demo dataset.

pub mod dataset;
pub mod features;
pub mod metrics;
pub mod report;
pub mod search;
pub mod synth;
