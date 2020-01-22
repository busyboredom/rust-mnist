extern crate rust_mnist;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rust_mnist::Mnist;
use std::time::Duration;

fn dataset_iter(c: &mut Criterion) {
    let mut custom = c.benchmark_group("Iterating Through Loaded MNIST");
    custom.measurement_time(Duration::from_secs(1));

    // Load the dataset into an "Mnist" object.
    let mnist = Mnist::new("examples/MNIST_data");

    custom.bench_function("Iterate Through Training Images", |b| {
        b.iter(|| {
            black_box({
                // Retrieve each image by iterating through them.
                for image in mnist.train_images() {
                    let _temp = image;
                }
            })
        })
    });

    custom.bench_function("Iterate Through Testing Images", |b| {
        b.iter(|| {
            black_box({
                // Retrieve each image by iterating through them.
                for image in mnist.test_images() {
                    let _temp = image;
                }
            })
        })
    });

    custom.bench_function("Iterate Through Training Labels", |b| {
        b.iter(|| {
            black_box({
                // Retrieve each label by iterating through them.
                for label in mnist.train_labels() {
                    let _temp = label;
                }
            })
        })
    });

    custom.bench_function("Iterate Through Testing Labels", |b| {
        b.iter(|| {
            black_box({
                // Retrieve each label by iterating through them.
                for label in mnist.test_labels() {
                    let _temp = label;
                }
            })
        })
    });

    custom.bench_function("Iterate Through Training Set", |b| {
        b.iter(|| {
            black_box({
                // Retrieve each pair by iterating through them.
                for pair in mnist.train_set() {
                    let _temp = pair;
                }
            })
        })
    });

    custom.bench_function("Iterate Through Testing Set", |b| {
        b.iter(|| {
            black_box({
                // Retrieve each pair by iterating through them.
                for pair in mnist.test_set() {
                    let _temp = pair;
                }
            })
        })
    });
}

criterion_group!(benches, dataset_iter);
criterion_main!(benches);
