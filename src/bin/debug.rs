//! # Debug testing environment

// -----------------------------------------------------------------------------------------------
// IMPORTS
// -----------------------------------------------------------------------------------------------

use image::RgbImage;
use imageproc::drawing;
use cv_feature::{
    fast::{Fast, Variant, KeyPoint},
    FeatureDetector
};
use minifb::{Window, WindowOptions};

// -----------------------------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    
    let img = image::open("res/westminster.jpg")?;
    let mut rgb_img = img.to_rgb();

    let width = rgb_img.width() as usize;
    let height = rgb_img.height() as usize;

    // Buffer for window
    let mut buffer: Vec<u32> = vec![0; width * height];

    let mut window = Window::new(
        "orbrs debug",
        width, height,
        WindowOptions::default()
    ).unwrap();

    window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));

    let fast = Fast::new(Variant::Fast9, 1.0/255.0*15.0);

    let keypoints = fast.extract(&img).0;

    println!("Found {} keypoints", keypoints.len());

    draw_keypoints(&mut rgb_img, keypoints);

    while window.is_open() {

        write_rgb_img_to_buffer(&rgb_img, width, height, &mut buffer);

        window
            .update_with_buffer(&buffer, width, height)
            .unwrap();
    }

    Ok(())
}

fn write_rgb_img_to_buffer(img: &RgbImage, width: usize, height: usize, buffer: &mut Vec<u32>) {
    for y in 0..height {
        for x in 0..width {
            buffer[x + (y * width)] = rgb_to_rgba_u32(img.get_pixel(x as u32, y as u32));
        }
    }
}

#[inline]
fn rgb_to_rgba_u32(rgb: &image::Rgb<u8>) -> u32 {
    255u32 << 24 | (rgb.0[0] as u32) << 16 | (rgb.0[1] as u32) << 8 | rgb.0[2] as u32
}

fn draw_keypoints(img: &mut RgbImage, keypoints: Vec<KeyPoint>) {
    // Color
    let col = image::Rgb::<u8>([0, 255, 0]);

    for kp in keypoints {
        drawing::draw_cross_mut(img, col, kp.point.x as i32, kp.point.y as i32);
    }
}