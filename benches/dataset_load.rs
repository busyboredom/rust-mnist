extern crate rust_mnist;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rust_mnist::Mnist;
use std::time::Duration;

fn load_dataset(c: &mut Criterion) {
    let mut custom = c.benchmark_group("Load MNIST");
    custom
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));

    custom.bench_function("Load Default Mnist", |b| {
        b.iter(|| {
            black_box({
                // Load the dataset.
                let _mnist = Mnist::new("examples/MNIST_data");
            })
        })
    });
}

criterion_group!(benches, load_dataset);
criterion_main!(benches);
