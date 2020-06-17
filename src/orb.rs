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

pub struct KeyPoint {
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
    type KeyPoint = KeyPoint;
    type Descriptor = BitArray<256>;

    fn extract(&self, _img: &DynamicImage) -> (Vec<Self::KeyPoint>, Vec<Self::Descriptor>) {
        unimplemented!();
    }
}

impl ImagePoint for KeyPoint {
    fn image_point(&self) -> Point2<f64> {
        self.point
    }
}
