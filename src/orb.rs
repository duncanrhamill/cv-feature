//! # ORB Feature Detector

// -----------------------------------------------------------------------------------------------
// IMPORTS
// -----------------------------------------------------------------------------------------------

use cv::{BitArray, ImagePoint};
use nalgebra::Point2;
use image::DynamicImage;
use crate::FeatureDetector;

// -----------------------------------------------------------------------------------------------
// DATA STRUCTURES
// -----------------------------------------------------------------------------------------------

/// An ORB (Oriented FAST and Rotated BRIEF) feature detector
pub struct Orb {

}

/// Represents a feature found by the [`Orb`] feature detector
pub struct Feature {

    /// The location of the feature within the image.
    pub point: Point2<f64>,

    /// A binary descriptor for the feature
    pub descriptor: BitArray<256>
}

// -----------------------------------------------------------------------------------------------
// IMPLEMENTATIONS
// -----------------------------------------------------------------------------------------------

impl Orb {
    /// Create a new instance of the ORB detector
    pub fn new() -> Self {
        Orb {}
    }
}

impl FeatureDetector for Orb {
    type Feature = Feature;

    fn extract(&self, _img: &DynamicImage) -> Vec<Self::Feature> {
        unimplemented!();
    }
}

impl ImagePoint for Feature {
    fn image_point(&self) -> Point2<f64> {
        self.point
    }
}
