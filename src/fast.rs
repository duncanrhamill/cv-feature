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

#[cfg(feature = "parallel")]
use {
    threadpool::ThreadPool,
    std::sync::{Arc, mpsc::channel},
    std::sync::atomic::{AtomicUsize, Ordering}
};

// -----------------------------------------------------------------------------------------------
// DATA STRUCTURES
// -----------------------------------------------------------------------------------------------

/// A FAST feature detector.
pub struct Fast {
    /// The variant of the FAST algorithm to use
    variant: Variant,

    /// The threshold value which N contiguous pixels must surpass before the central pixel is 
    /// considered a corner.
    threshold: f32,

    /// Displacement vectors for pixels in the ring around the central pixel.
    disp_vectors: [Vector2<i32>; 16],

    /// Thread pool to perform parallel computation if the parallel feature is enabled.
    #[cfg(feature = "parallel")]
    thread_pool: Option<ThreadPool>
}

/// A feature detected by the FAST algorithm
#[derive(Copy, Clone)]
pub struct Feature {

    /// The location of the feature within the image.
    pub point: Point2<usize>
}

// -----------------------------------------------------------------------------------------------
// ENUMERATIONS
// -----------------------------------------------------------------------------------------------

/// Possible variants of the FAST algorithm.
#[derive(Copy, Clone)]
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
    #[cfg(not(feature = "parallel"))]
    pub fn new(variant: Variant, threshold: f32) -> Self {
        Self {
            variant,
            threshold,
            disp_vectors: get_disp_vector_array()
        }
    }

    /// Create a new FAST detector
    ///
    /// # Arguments
    /// - `variant` - The type of FAST algorithm to use
    /// - `threshold` - The value of the threshold which N contiguous pixels must meet before the
    /// central pixel is considered a corner
    #[cfg(feature = "parallel")]
    pub fn new(variant: Variant, threshold: f32) -> Self {
        Self {
            variant,
            threshold,
            disp_vectors: get_disp_vector_array(),
            thread_pool: None
        }
    }

    /// Create a new FAST detector which will execute in parallel mode
    ///
    /// # Arguments
    /// - `variant` - The type of FAST algorithm to use
    /// - `threshold` - The value of the threshold which N contiguous pixels must meet before the
    /// central pixel is considered a corner
    /// - `num_threads` - The number of worker threads to use
    #[cfg(feature = "parallel")]
    pub fn new_parallel(variant: Variant, threshold: f32, num_threads: usize) -> Self {
        Self {
            variant,
            threshold,
            disp_vectors: get_disp_vector_array(),
            thread_pool: Some(ThreadPool::new(num_threads))
        }
    }

    pub(crate) fn extract_from_grey(&self, img: &GrayFloatImage) -> Vec<Feature> {
        // Vector of keypoints to return
        let mut keypoints = Vec::<Feature>::new();

        // Iterate through pixels in the image, ignoring the outer edges since the ring would go
        // outside the image bounds.
        for y in 3..(img.height() - 3) {
            for x in 3..(img.width() - 3) {
                // Get the keypoint based on the correct test for this variant of the algorithm
                let kp = match self.variant {
                    Variant::Fast12HighSpeed => unimplemented!(),
                    _ => segment_test_fast(
                        &img, 
                        Point2::from([x, y]),
                        &self.variant,
                        &self.disp_vectors,
                        self.threshold
                    )
                };

                // Add the keypoint or ignore it if none is found
                match kp {
                    Some(kp) => keypoints.push(kp),
                    None => ()
                }
            }
        }

        // Return the keypoints
        keypoints
    }

    fn extract_st(&self, img: &DynamicImage) -> Vec<Feature> {
        // Get the float image
        let float_image = GrayFloatImage::from_dynamic(img);
        
        self.extract_from_grey(&float_image)
    }
}

impl FeatureDetector for Fast {
    type Feature = Feature;

    /// Extract keypoints from the provided image
    #[cfg(not(feature = "parallel"))]
    fn extract(&self, img: &DynamicImage) -> Vec<Self::Feature> {
        self.extract_st(img)
    }


    /// Extract keypoints from the provided image
    ///
    /// If the `Fast` instance was initialised using `new_parallel` the work will be performed in
    /// a multi-threaded manner. 
    ///
    /// TODO: this currently runs very slowly, probably because we have a separate job per pixel.
    /// Maybe we should consider splitting the image into segments and processing each in parallel.
    #[cfg(feature = "parallel")]
    fn extract(&self, img: &DynamicImage) -> Vec<Self::Feature> {
        
        // Branch depending on whether or not we have a threadpool to work with
        match self.thread_pool {
            Some(ref tp) => {
                // Extract the gray image
                let float_image = Arc::new(
                    GrayFloatImage::from_dynamic(img)
                );

                // Create new owned bits of self, because we can't move self into the closure
                let variant = self.variant;
                let disp_vectors = self.disp_vectors;
                let threshold = self.threshold;

                // Create chanels to send keypoints back and forth
                let (tx, rx) = channel();

                // Counter for number of produced keypoints
                let num_kps = Arc::new(AtomicUsize::new(0));

                // Iterate over pixels, adding workers to the pool
                for y in 3..(float_image.height() - 3) {
                    for x in 3..(float_image.width() - 3) {
                        // Clone the image and keypoint vector
                        let img = float_image.clone();
                        let tx = tx.clone();
                        let nkps = num_kps.clone();
                        tp.execute(move || {
                            match segment_test(
                                &img, 
                                Point2::from([x, y]), 
                                &variant, 
                                &disp_vectors, 
                                threshold
                            ) {
                                Some(kp) => {
                                    tx.send(kp).unwrap();
                                    nkps.fetch_add(1, Ordering::Relaxed);
                                },
                                None => ()
                            };
                        });
                    }
                }
                
                // Join threads
                tp.join();

                // Collect the keypoints from the reciever
                let keypoints = rx.iter().take(
                    num_kps.load(Ordering::Relaxed)
                ).collect();

                keypoints
            },
            // If no threadpool revert to the single threaded version
            None => {
                self.extract_st(img)
            }
        }
    }
}

impl ImagePoint for Feature {
    fn image_point(&self) -> Point2<f64> {
        Point2::from([self.point.x as f64, self.point.y as f64])
    }
}

// -----------------------------------------------------------------------------------------------
// PRIVATE FUNCTIONS
// -----------------------------------------------------------------------------------------------

/// Get the array of displacment vectors.
fn get_disp_vector_array() -> [Vector2<i32>; 16] {
    [
        Vector2::from([ 0,  3]),
        Vector2::from([ 1,  3]),
        Vector2::from([ 2,  2]),
        Vector2::from([ 3,  1]),
        Vector2::from([ 3,  0]),
        Vector2::from([ 3, -1]),
        Vector2::from([ 2, -2]),
        Vector2::from([ 1, -3]),
        Vector2::from([ 0, -3]),
        Vector2::from([-1, -3]),
        Vector2::from([-2, -2]),
        Vector2::from([-3, -1]),
        Vector2::from([-3,  0]),
        Vector2::from([-3,  1]),
        Vector2::from([-2,  2]),
        Vector2::from([-1,  3])
    ]
}

/// Perform the segment test on the image at the given point.
///
/// Returns a `Feature` if the pixel is a feature and `None` otherwise.
///
/// This function performs no checking on `point` being within the valid image region in which
/// the FAST ring will not overlap the edge of the image. This check must be performed at the
/// higher level.
///
/// This function is implemented outside the `Fast` object so that it may be used in parallel 
/// thread execution without needing to send `self`.
///
/// # Arguments
/// - `img` - A reference to the `GrayFloatImage` to operate on
/// - `point` - The point to test
/// - `variant` - A reference to the variant of the test to use, FAST-9 or FAST-12
/// - `disp_vectors` - A reference to the displacement vectors which define the shape of the test
///    ring
/// - `threshold` - The threshold to test for.
fn segment_test(
    img: &GrayFloatImage, 
    point: Point2<usize>, 
    variant: &Variant, 
    disp_vectors: &[Vector2<i32>; 16], 
    threshold: f32
) -> Option<Feature> {
    // Get the number of required contiguous pixels based on the variant
    let num_required = match variant {
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
        let disp = disp_vectors[i];

        // Get the intensity of this pixel
        let intensity = img.get(
            ((point.x as i32) + disp.x) as usize, 
            ((point.y as i32) + disp.y) as usize
        );

        // Set the condition for this pixel (no need to set it if there is no condition)
        if intensity > central_intensity + threshold {
            ring_conditions[i] = Condition::Brighter;
        }
        else if intensity < central_intensity - threshold {
            ring_conditions[i] = Condition::Dimmer;
        }
    }

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
        Some(Feature {
            point
        })
    }
    // Otherwise return None since there's no keypoint here
    else {
        None
    }
}

/// Perform a faster segment test on the image at the given point.
///
/// Returns a `Feature` if the pixel is a feature and `None` otherwise.
///
/// This function performs no checking on `point` being within the valid image region in which
/// the FAST ring will not overlap the edge of the image. This check must be performed at the
/// higher level.
///
/// This function is implemented outside the `Fast` object so that it may be used in parallel 
/// thread execution without needing to send `self`.
///
/// This faster segment test aims to exit the test with `None` at the earliest possible point. This
/// is done by remembering if there is a region in the first 
///
/// # Arguments
/// - `img` - A reference to the `GrayFloatImage` to operate on
/// - `point` - The point to test
/// - `variant` - A reference to the variant of the test to use, FAST-9 or FAST-12
/// - `disp_vectors` - A reference to the displacement vectors which define the shape of the test
///    ring
/// - `threshold` - The threshold to test for.
fn segment_test_fast(
    img: &GrayFloatImage, 
    point: Point2<usize>, 
    variant: &Variant, 
    disp_vectors: &[Vector2<i32>; 16], 
    threshold: f32
) -> Option<Feature> {
    // Get the number of required contiguous pixels based on the variant
    let region_target = match variant {
        Variant::Fast9 => 9,
        Variant::Fast12 => 12,
        _ => panic!("Non-high-speed test used in FAST-X High Speed algorithm")
    };

    // Counters for the running and first region length
    let mut running_len = 0;
    let mut first_len = 0;

    // Conditions for the running and first regions
    let mut running_cond = Condition::None;
    let mut first_cond = Condition::None;

    // Index at which to switch to checking against first condition. Starts assuming there's no
    // first region.
    let mut pivot_index = 16 - region_target;

    // Indensity of the central pixel
    let central_intensity = img.get(point.x, point.y);

    // Loop over the ring
    for i in 0..16 {
        // Condition for this pixel
        let mut cond = Condition::None;
        
        // Displacement vector for this pixel
        let disp = disp_vectors[i];

        // Get the intensity of this pixel
        let intensity = img.get(
            ((point.x as i32) + disp.x) as usize, 
            ((point.y as i32) + disp.y) as usize
        );

        // Set the condition for this pixel (no need to set it if there is no condition)
        if intensity > central_intensity + threshold {
            cond = Condition::Brighter;
        }
        else if intensity < central_intensity - threshold {
            cond = Condition::Dimmer;
        }

        // If the current pixel is a different condition than the running one
        if cond != running_cond {
            // If the current index is greater than the pivot, and the condition is different than
            // the first there is no way we can build a region of the required number of pixels
            // so this is not a feature.
            if i > pivot_index && cond != first_cond {
                return None;
            }

            // Otherwise update the running condition to the current one and reset the length
            running_cond = cond;
            running_len = 0;

            // If the current condition is either dimmer of brighter then increment the running 
            // length
            if cond != Condition::None {
                running_len += 1;
            }
        }
        // If the current pixel is the same condition (but not none) as the running increment the 
        // running length
        else if cond != Condition::None {
            running_len += 1;
        }

        // If the current index is the same as the running length then we are in the first region
        // so set the first region data (have to increment i since regions are length-based)
        if i + 1 == running_len {
            first_cond = cond;
            first_len = running_len;
            
            // Change the pivot index to account for the increased first region length
            pivot_index = 16 - region_target + first_len;
        }

        // If the running region length is equal to the target then return the found feature
        if running_len >= region_target {
            return Some(Feature {
                point
            })
        }
    }

    // At the end of the loop if the sum of the first region and the running region is greater than
    // the target we have found a region
    if running_len + first_len >= region_target {
        Some(Feature {
            point
        })
    }
    else {
        None
    }
}