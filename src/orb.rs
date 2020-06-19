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

pub struct Orb {

}

pub struct Feature {
    point: Point2<f64>
}

// -----------------------------------------------------------------------------------------------
// IMPLEMENTATIONS
// -----------------------------------------------------------------------------------------------

impl Orb {
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
