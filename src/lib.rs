//! # Rust-CV feature detectors
//! 
//! This is a test implementation designed to be used with [`rust-cv`](https://github.com/rust-cv),
//! as an alternative to the AKAZE implementation. The goal is to implement ORB.
//!
//! This crate provides the `FeatureDetector` trait which abstracts away different detectors, 
//! allowing them to be used interchangably. The trait is also implemented for the `Akaze` detector.

// -----------------------------------------------------------------------------------------------
// IMPORTS
// -----------------------------------------------------------------------------------------------

use ::image::DynamicImage;
use cv::BitArray;
use cv::ImagePoint;
use cv::feature::akaze::{Akaze, KeyPoint};

// -----------------------------------------------------------------------------------------------
// MODULES
// -----------------------------------------------------------------------------------------------

pub mod fast;
pub mod orb;
mod image;

// -----------------------------------------------------------------------------------------------
// TRAITS
// -----------------------------------------------------------------------------------------------

pub trait FeatureDetector {
    type KeyPoint: ImagePoint;
    type Descriptor;

    /// Extract features from an image.
    fn extract(&self, img: &DynamicImage) -> (Vec<Self::KeyPoint>, Vec<Self::Descriptor>);
}

// -----------------------------------------------------------------------------------------------
// IMPLEMENTATIONS
// -----------------------------------------------------------------------------------------------

impl FeatureDetector for Akaze {
    type KeyPoint = KeyPoint;
    type Descriptor = BitArray<64>;

    fn extract(&self, img: &DynamicImage) -> (Vec<Self::KeyPoint>, Vec<Self::Descriptor>) {
        self.extract(img)
    }
}
