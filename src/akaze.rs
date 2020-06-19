//! # AKAZE feature detector implementation
//!
//! This module simpl implements the `FeatureDetector` trait for the already implemented `Akaze`
//! struct from `cv::feature::akaze`.

// -----------------------------------------------------------------------------------------------
// IMPORTS
// -----------------------------------------------------------------------------------------------

use crate::FeatureDetector;
use image::DynamicImage;
use nalgebra::Point2;
use cv::feature::akaze::{Akaze, KeyPoint};
use cv::{BitArray, ImagePoint};

// -----------------------------------------------------------------------------------------------
// DATA STRUCTURES
// -----------------------------------------------------------------------------------------------

/// A feature detected by the AKAZE algorithm.
pub struct Feature {
    /// The AKAZE keypoint
    pub keypoint: KeyPoint,

    /// A descriptor for the feature
    pub descriptor: BitArray<64>
}

// -----------------------------------------------------------------------------------------------
// IMPLEMENTATIONS
// -----------------------------------------------------------------------------------------------

impl FeatureDetector for Akaze {
    type Feature = Feature;

    fn extract(&self, img: &DynamicImage) -> Vec<Self::Feature> {
        let (kps, descs) = self.extract(img);

        kps.iter().zip(descs).map(|(k, d)| Feature {
            keypoint: *k,
            descriptor: d
        }).collect()
    }
}

impl ImagePoint for Feature {
    fn image_point(&self) -> Point2<f64> {
        self.keypoint.image_point()
    }
}