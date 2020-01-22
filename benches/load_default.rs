extern crate rust_mnist;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rust_mnist::Mnist;

fn load_default(c: &mut Criterion) {
    let mut custom = c.benchmark_group("Load MNIST");
    custom.sample_size(10);
    // Load the dataset into an "Mnist" object.
    custom.bench_function("Load Default Mnist", |b| {
        b.iter(|| {
            black_box({
                let _mnist = Mnist::new("examples/MNIST_data");
            })
        })
    });
}

criterion_group!(benches, load_default);
criterion_main!(benches);
