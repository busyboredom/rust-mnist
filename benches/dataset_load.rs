extern crate rust_mnist;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rust_mnist::Mnist;
use std::time::Duration;

fn load_dataset(c: &mut Criterion) {
    let mut custom = c.benchmark_group("Load MNIST");
    custom
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));

    // TODO: Make windows compatible.
    custom.bench_function("Load Default Mnist", |b| {
        b.iter(|| {
            // Load the dataset.
            let mnist = Mnist::new("examples/MNIST_data/");
            black_box(mnist);
        })
    });
}

criterion_group!(benches, load_dataset);
criterion_main!(benches);
