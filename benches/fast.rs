//! # FAST benchmarks

use cv_feature::{
    fast::{Fast, Variant},
    FeatureDetector
};
use image;
use criterion::{criterion_group, criterion_main, Criterion};
use imageproc::corners::corners_fast9;

fn bench_westminster(c: &mut Criterion) {
    let img = image::open("res/westminster.jpg").unwrap();

    let fast = Fast::new(Variant::Fast9, 0.1);

    c.bench_function("Fast9::extract", |b| b.iter(|| fast.extract(&img)));
}

fn bench_westminster_imgproc(c: &mut Criterion) {
    let img = image::open("res/westminster.jpg").unwrap().into_luma();

    c.bench_function("corners_fast9", |b| b.iter(|| corners_fast9(&img, 26)));
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_westminster, bench_westminster_imgproc
);
criterion_main!(benches);