//! # ORB Feature Detector

// -----------------------------------------------------------------------------------------------
// IMPORTS
// -----------------------------------------------------------------------------------------------

use cv::{BitArray, ImagePoint};
use nalgebra::Point2;
use image::DynamicImage;
use crate::{FeatureDetector, image::GrayFloatImage, fast::{Fast, Variant, Feature as FastFeature}};

// -----------------------------------------------------------------------------------------------
// DATA STRUCTURES
// -----------------------------------------------------------------------------------------------

/// An ORB (Oriented FAST and Rotated BRIEF) feature detector
pub struct Orb {
    /// The number of levels to use in the scale space pyramid.
    num_levels: usize,

    /// Threshold to use for the FAST detector. 
    fast_threshold: f32
}

/// Represents a feature found by the [`Orb`] feature detector
pub struct Feature {

    /// The location of the feature within the image.
    pub point: Point2<f64>,

    /// A binary descriptor for the feature
    pub descriptor: BitArray<256>
}

/// A scale-pyramid for a particular image.
struct ScalePyramid(Vec<GrayFloatImage>);

/// An interator over the levels of a scale space pyramid.
struct ScalePyramidIter<'a> {
    /// The pyramid to iterate over
    pyramid: &'a ScalePyramid,

    /// The current index of the iterator
    index: usize
}

// -----------------------------------------------------------------------------------------------
// IMPLEMENTATIONS
// -----------------------------------------------------------------------------------------------

impl Orb {
    
}

impl Default for Orb {
    fn default() -> Self {
        Self {
            num_levels: 4,
            fast_threshold: 0.1
        }
    }
}

impl FeatureDetector for Orb {
    type Feature = Feature;

    fn extract(&self, img: &DynamicImage) -> Vec<Self::Feature> {

        // ---- FAST ----

        // Start by constructing the scale pyramid for this image
        let pyramid = ScalePyramid::construct(img, self.num_levels);

        let mut fast_features = Vec::with_capacity(self.num_levels);

        // FAST detector instance
        let fast = Fast::new(Variant::Fast9, self.fast_threshold);
        
        // Run FAST over each item
        for level in pyramid.iter() {
            fast_features.push(fast.extract_from_grey(level))
        }

        unimplemented!()
    }
}

impl ImagePoint for Feature {
    fn image_point(&self) -> Point2<f64> {
        self.point
    }
}

impl ScalePyramid {
    
    /// Construct a pyramid from an input image with the given number of levels.
    fn construct(img: &DynamicImage, num_levels: usize) -> Self {
        let grey = GrayFloatImage::from_dynamic(img);

        let mut levels = vec![grey];

        for i in 1..num_levels {
            levels.push(levels[i - 1].half_size());
        }

        Self(levels)
    }

    fn iter<'a>(&'a self) -> ScalePyramidIter<'a> {
        ScalePyramidIter {
            pyramid: self,
            index: 0
        }
    }
}

impl<'a> Iterator for ScalePyramidIter<'a> {
    type Item = &'a GrayFloatImage;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.pyramid.0.len() - 1 {
            None
        }
        else {
            self.index += 1;
            Some(&self.pyramid.0[self.index])
        }
    }
}
