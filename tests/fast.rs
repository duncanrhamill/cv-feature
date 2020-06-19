//! # Basic functionality tests for FAST

// -----------------------------------------------------------------------------------------------
// IMPORTS
// -----------------------------------------------------------------------------------------------

use cv::ImagePoint;
use cv_feature::{
    fast::{Fast, Variant},
    FeatureDetector
};
use nalgebra::Point2;
use image;

// -----------------------------------------------------------------------------------------------
// TYPES
// -----------------------------------------------------------------------------------------------

type Result = std::result::Result<(), Box<dyn std::error::Error>>;

// -----------------------------------------------------------------------------------------------
// TESTS
// -----------------------------------------------------------------------------------------------

/// Test the basic implementation, run FAST on the westminster image.
#[test]
fn westminster() -> Result {
    let img = image::open("res/westminster.jpg")?;

    let fast = Fast::new(Variant::Fast9, 0.1);

    let features = fast.extract(&img);

    assert!(features.len() > 0);

    Ok(())
}

/// Test that the algorithm will find regions at the top of the ring (which straddle the start)
#[test]
fn region_at_top() -> Result {
    let img = image::open("res/small_top_region.png")?;

    let fast = Fast::new(Variant::Fast9, 0.1);

    let features = fast.extract(&img);

    assert!(features.len() == 1);

    Ok(())
}

/// Test this implementation against the OpenCV one using some test vectors.
/// 
/// We are not looking for exact matches between them, as this is unlikely since there are 
/// different approaches used between OpenCV and this implementation. Instead we check for:
/// - the number of matched features being similar 
/// - the mean of matched x coordinates being similar 
/// - the mean of matched y coordinates being similar 
///
/// Here "similar" is considered to be 1.25% difference.
#[test]
fn test_vectors() -> Result {

    // Similarity threshold
    const SIMILAR_THRESHOLD: f64 = 0.0125;

    let img = image::open("res/westminster.jpg")?;

    // Use approximately the same threshold as in the gen_fast_tv.py file, 51.
    let fast = Fast::new(Variant::Fast9, 1.0/255.0*51.0);

    let features = fast.extract(&img);

    // Load the test vector CSV
    let mut reader = csv::Reader::from_path(
        "test_vectors/fast_test_vectors.csv"
    )?;

    let mut test_vector: Vec<Point2<f64>> = Vec::new();

    for line in reader.records() {

        let line = line?;

        test_vector.push(
            Point2::new(line.get(0).unwrap().parse()?, line.get(1).unwrap().parse()?)
        );
    }

    // Check that the number of matches is similar
    if rel_diff(test_vector.len() as f64, features.len() as f64) > SIMILAR_THRESHOLD
    {
        panic!(
            "Number of matches are dissimilar ({} for test vectors vs {})", 
            test_vector.len(), features.len()
        );    
    }

    // Compute the mean of x and y coordinates
    let (mut accum_x, mut accum_y) = (0.0, 0.0);

    for kp in &features {
        accum_x += kp.image_point().x;
        accum_y += kp.image_point().y;
    }
    let mean_x_kp = accum_x / features.len() as f64;
    let mean_y_kp = accum_y / features.len() as f64;

    accum_x = 0.0;
    accum_y = 0.0;

    for tv in &test_vector {
        accum_x += tv.x;
        accum_y += tv.y;
    }
    let mean_x_tv = accum_x / test_vector.len() as f64;
    let mean_y_tv = accum_y / test_vector.len() as f64;

    // Compare x and y means
    if rel_diff(mean_x_kp, mean_x_tv) > SIMILAR_THRESHOLD
    {
        panic!(
            "Mean x coordinates are dissimilar ({} for test vectors vs {}", mean_x_tv, mean_x_kp
        );
    }

    if rel_diff(mean_y_kp, mean_y_tv) > SIMILAR_THRESHOLD
    {
        panic!(
            "Mean y coordinates are dissimilar ({} for test vectors vs {}", mean_y_tv, mean_y_kp
        );
    }

    Ok(())
}

fn rel_diff(a: f64, b: f64) -> f64 {
    (a - b).abs() / a.max(b)
}