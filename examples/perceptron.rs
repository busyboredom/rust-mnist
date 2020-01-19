extern crate rand; // For initializing weights.
extern crate rust_mnist;

use rand::distributions::{IndependentSample, Range};
use rust_mnist::{print_sample_image, Mnist};

// Hyperparameter
const LEARNING_RATE: f64 = 0.0001;
const BIAS: f64 = 1.0;

fn main() {
    // Load the dataset into an "Mnist" object.
    let mnist = Mnist::new("examples/MNIST_data");

    // Print one image for verification.
    print_sample_image(mnist.get_train_image(2), mnist.get_train_label(2));

    // Generate an array of random weights.
    let mut weights = generate_weights();

    // Training.
    let mut accuracy = 0.0;
    for iter in 0..5 {
        for image_index in 0..60_000 {
            print!("Epoch: {:2}  Iter: {:5}  ", iter, image_index);

            // Get an image.
            let image = normalize(mnist.get_train_image(image_index));

            // Get label.
            let label = mnist.get_train_label(image_index);

            // Calculate the outputs.
            let mut outputs = dot_product(&image, weights);
            outputs = softmax(&outputs);

            // Calculate the error.
            let mut error: [f64; 10] = subtract(outputs, one_hot(label));

            // Update rolling-average accuracy.
            accuracy = {
                (accuracy * 999.0 + {
                    if largest(&outputs) == usize::from(label) {
                        1.0
                    } else {
                        0.0
                    }
                }) / 1000.0
            };
            println!("Accuracy: {:.2}", accuracy);

            // Update weights.
            update(&mut weights, &error, &image);
        }
    }
}

fn update(weights: &mut [[f64; 785]; 10], error: &[f64; 10], image: &Vec<f64>) {
    for class_index in 0..error.len() {
        for input_index in 0..image.len() {
            weights[class_index][input_index] -=
                LEARNING_RATE * error[class_index] * image[input_index];
            weights[class_index][784] -= LEARNING_RATE * error[class_index] * BIAS;
        }
    }
}

fn generate_weights() -> [[f64; 785]; 10] {
    // Preparing the random number generator before initializing weights.
    let range = Range::new(0.0, 1.0);
    let mut rng = rand::thread_rng();

    // Creating a weight array.
    let mut weights: [[f64; 785]; 10] = [[0.0; 785]; 10];

    // Initializing the weights.
    for class_weights in weights.iter_mut() {
        for weight in class_weights.iter_mut() {
            *weight = range.ind_sample(&mut rng);
        }
    }
    weights
}

fn dot_product(image: &Vec<f64>, weights: [[f64; 785]; 10]) -> [f64; 10] {
    let mut outputs: [f64; 10] = [0.0; 10];
    for output_index in 0..outputs.len() {
        for pixel_index in 0..image.len() {
            outputs[output_index] +=
                f64::from(image[pixel_index]) * weights[output_index][pixel_index];
            outputs[output_index] += BIAS * weights[output_index][784];
        }
    }
    outputs
}

fn subtract(lhs: [f64; 10], rhs: [f64; 10]) -> [f64; 10] {
    let mut difference: [f64; 10] = [0.0; 10];
    for index in 0..difference.len() {
        difference[index] = lhs[index] - rhs[index];
    }
    difference
}

fn one_hot(value: u8) -> [f64; 10] {
    let mut arr: [f64; 10] = [0.0; 10];
    arr[usize::from(value)] = 1.0;
    arr
}

fn normalize(image: &Vec<u8>) -> Vec<f64> {
    // Normalize the image.
    image
        .into_iter()
        .map(|pixel| 2.0 * f64::from(*pixel) / 255.0 - 1.0)
        .collect()
}

fn largest(arr: &[f64; 10]) -> usize {
    arr.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index)
        .unwrap()
}

fn softmax(arr: &[f64; 10]) -> [f64; 10] {
    let exp: Vec<f64> = arr.iter().map(|x| x.exp()).collect();
    let sum_exp: f64 = exp.iter().sum();
    let mut softmax_arr: [f64; 10] = [0.0; 10];
    for index in 0..softmax_arr.len() {
        softmax_arr[index] = exp[index] / sum_exp;
    }
    softmax_arr
}
