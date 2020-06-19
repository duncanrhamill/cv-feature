//! # Rust-CV feature detectors
//! 
//! This is a test implementation designed to be used with [`rust-cv`](https://github.com/rust-cv),
//! as an alternative to the AKAZE implementation. The goal is to implement ORB.
//!
//! This crate provides the `FeatureDetector` trait which abstracts away different detectors, 
//! allowing them to be used interchangably. The trait is also implemented for the `Akaze` detector.

#![warn(missing_docs)]

// -----------------------------------------------------------------------------------------------
// IMPORTS
// -----------------------------------------------------------------------------------------------

use ::image::DynamicImage;
use cv::ImagePoint;

// -----------------------------------------------------------------------------------------------
// MODULES
// -----------------------------------------------------------------------------------------------

pub mod akaze;
pub mod fast;
pub mod orb;
mod image;

// -----------------------------------------------------------------------------------------------
// TRAITS
// -----------------------------------------------------------------------------------------------

/// Trait for types which act as feature detectors.
pub trait FeatureDetector {

    /// The feature type returned by this detector.
    ///
    /// The chosen type should implement the [`cv::ImagePoint`] trait.
    type Feature: ImagePoint;

    /// Extract features from an image.
    fn extract(&self, img: &DynamicImage) -> Vec<Self::Feature>;
}
