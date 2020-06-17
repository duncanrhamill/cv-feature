//! # A FAST feature detector
//!
//! This implementation of FAST is used by the ORB detector [`Orb`].

// -----------------------------------------------------------------------------------------------
// IMPORTS
// -----------------------------------------------------------------------------------------------

use ::image::DynamicImage;
use nalgebra::{Vector2, Point2};
use cv::ImagePoint;
use crate::image::GrayFloatImage;
use crate::FeatureDetector;

// -----------------------------------------------------------------------------------------------
// DATA STRUCTURES
// -----------------------------------------------------------------------------------------------

pub struct Fast {
    /// The variant of the FAST algorithm to use
    variant: Variant,

    /// The threshold value which N contiguous pixels must surpass before the central pixel is 
    /// considered a corner.
    threshold: f32
}

pub struct KeyPoint {
    pub point: Point2<usize>
}

// -----------------------------------------------------------------------------------------------
// ENUMERATIONS
// -----------------------------------------------------------------------------------------------

/// Possible variants of the FAST algorithm.
pub enum Variant {
    /// FAST-9, in which N (number of contiguous pixels in the ring) is 9.
    Fast9,

    /// FAST-12, in which N (number of contiguous pixels in the ring) is 12.
    Fast12,

    /// FAST-12 with High Speed Test, an optimsation over FAST-12 described 
    /// [here](https://en.wikipedia.org/wiki/Features_from_accelerated_segment_test#High-speed_test)
    Fast12HighSpeed,
}

/// Condition we are checking for in the segment test
#[derive(Eq, PartialEq, Copy, Clone, Debug)]
enum Condition {
    Brighter,
    Dimmer,
    None
}

// -----------------------------------------------------------------------------------------------
// IMPLEMENTATIONS
// -----------------------------------------------------------------------------------------------

impl Fast {

    /// Create a new FAST detector
    ///
    /// # Arguments
    /// - `variant` - The type of FAST algorithm to use
    /// - `threshold` - The value of the threshold which N contiguous pixels must meet before the
    /// central pixel is considered a corner
    pub fn new(variant: Variant, threshold: f32) -> Self {
        Self {
            variant,
            threshold
        }
    }

    /// Test a particular pixel for being a feature.
    ///
    /// Returns a `KeyPoint` if the pixel is a feature and `None` otherwise.
    ///
    /// This function performs no checking on `point` being within the valid image region in which
    /// the FAST ring will not overlap the edge of the image. This check must be performed at the
    /// higher level.
    #[inline]
    fn segment_test(&self, img: &GrayFloatImage, point: Point2<usize>) -> Option<KeyPoint> {
        // Get the number of required contiguous pixels based on the variant
        let num_required = match self.variant {
            Variant::Fast9 => 9,
            Variant::Fast12 => 12,
            _ => panic!("Non-high-speed test used in FAST-X High Speed algorithm")
        };

        // Running counter for number of contiguous pixels meeting the threshold
        let mut num_contig = 0;

        // Condition to be checked for, is set by the first pixel to meet a condition
        let mut condition = Condition::None;

        // Intensity of the central pixel
        let central_intensity = img.get(point.x, point.y);

        // Read the intensities of the pixels in the ring
        let mut ring_conditions = [Condition::None; 16];
        for i in 0..16usize {
            // Displacement vector for this pixel
            let disp = circ_idx_to_disp_vec(i as i32).unwrap();

            // Get the intensity of this pixel
            let intensity = img.get(
                ((point.x as i32) + disp.x) as usize, 
                ((point.y as i32) + disp.y) as usize
            );

            // Set the condition for this pixel (no need to set it if there is no condition)
            if intensity > central_intensity + self.threshold {
                ring_conditions[i] = Condition::Brighter;
            }
            else if intensity < central_intensity - self.threshold {
                ring_conditions[i] = Condition::Dimmer;
            }
        }

        // println!("{:?}", ring_conditions);

        // Flag indicating if the pixel in the ring has already been visited
        let mut visited = [false; 16];

        // Pixel index
        let mut idx = 0;

        // Iterate through the ring until:
        //  - N contiguous pixels meeting one of the conditions are found
        //  - We reach a pixel that's already been visited when the running counter is less than 2.
        while (num_contig < num_required)
            &&
            ((visited[idx] && num_contig > 1) || !visited[idx])
        {
            // If we hit a new condition
            if condition != ring_conditions[idx] {
                // Update the running condition
                condition = ring_conditions[idx];

                // If the new condition is not none start counting a contigous region
                if ring_conditions[idx] != Condition::None {
                    num_contig = 1;
                }
                // If the new condition is None then there is no contigious region
                else {
                    num_contig = 0;
                }
            }
            // If this pixel meets a condition increment the counter
            else if ring_conditions[idx] != Condition::None {
                num_contig += 1;
            }

            // Set the pixel to visited
            visited[idx] = true;
            
            // println!("{}: {:?}", idx, visited);

            // Increment the index, wrapping at 16
            idx = (idx + 1) % 16;
        }

        // If we exited because we hit the condition return a new keypoint
        if num_contig >= num_required {
            Some(KeyPoint {
                point
            })
        }
        // Otherwise return None since there's no keypoint here
        else {
            None
        }
    }
}

impl FeatureDetector for Fast {
    type KeyPoint = KeyPoint;
    type Descriptor = ();

    /// Extract keypoints from the provided image
    fn extract(&self, img: &DynamicImage) -> (Vec<Self::KeyPoint>, Vec<Self::Descriptor>) {

        // Get the float image
        let float_image = GrayFloatImage::from_dynamic(img);
        
        // Vector of keypoints to return
        let mut keypoints = Vec::<KeyPoint>::new();

        // Iterate through pixels in the image, ignoring the outer edges since the ring would go
        // outside the image bounds.
        for y in 3..(float_image.height() - 3) {
            for x in 3..(float_image.width() - 3) {
                // Get the keypoint based on the correct test for this variant of the algorithm
                let kp = match self.variant {
                    Variant::Fast12HighSpeed => unimplemented!(),
                    _ => self.segment_test(&float_image, Point2::from([x, y]))
                };

                // Add the keypoint or ignore it if none is found
                match kp {
                    Some(kp) => keypoints.push(kp),
                    None => ()
                }
            }
        }

        // Return the keypoints
        (keypoints, vec![])
    }
}

impl ImagePoint for KeyPoint {
    fn image_point(&self) -> Point2<f64> {
        Point2::from([self.point.x as f64, self.point.y as f64])
    }
}

// -----------------------------------------------------------------------------------------------
// PRIVATE FUNCTIONS
// -----------------------------------------------------------------------------------------------

/// Convert a pixel index (zero-based) to a displacement vector from the central pixel.
///
/// Indexes not within the circle (`idx < 0 || idx > 15`) will return `None`.
#[inline]
fn circ_idx_to_disp_vec(idx: i32) -> Option<Vector2<i32>> {
    match idx {
         0 => Some(Vector2::from([ 0,  3])),
         1 => Some(Vector2::from([ 1,  3])),
         2 => Some(Vector2::from([ 2,  2])),
         3 => Some(Vector2::from([ 3,  1])),
         4 => Some(Vector2::from([ 3,  0])),
         5 => Some(Vector2::from([ 3, -1])),
         6 => Some(Vector2::from([ 2, -2])),
         7 => Some(Vector2::from([ 1, -3])),
         8 => Some(Vector2::from([ 0, -3])),
         9 => Some(Vector2::from([-1, -3])),
        10 => Some(Vector2::from([-2, -2])),
        11 => Some(Vector2::from([-3, -1])),
        12 => Some(Vector2::from([-3,  0])),
        13 => Some(Vector2::from([-3,  1])),
        14 => Some(Vector2::from([-2,  2])),
        15 => Some(Vector2::from([-1,  3])),
        _  => None
    }
}