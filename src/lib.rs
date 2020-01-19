// A simple class and associated functions for parsing the MNIST dataset.

use log::{debug, info};
use std::convert::TryFrom;
use std::fs;
use std::io;
use std::io::Read;

// Filenames
const TRAIN_DATA_FILENAME: &str = "/train-images-idx3-ubyte";
const TEST_DATA_FILENAME: &str = "/t10k-images-idx3-ubyte";
const TRAIN_LABEL_FILENAME: &str = "/train-labels-idx1-ubyte";
const TEST_LABEL_FILENAME: &str = "/t10k-labels-idx1-ubyte";

// Constants relating to the MNIST dataset.
const IMAGES_MAGIC_NUMBER: u32 = 2051;
const LABELS_MAGIC_NUMBER: u32 = 2049;
const NUM_TRAIN_IMAGES: u32 = 60_000;
const NUM_TEST_IMAGES: u32 = 10_000;
const IMAGE_ROWS: u32 = 28;
const IMAGE_COLUMNS: u32 = 28;

pub struct Mnist {
    train_data: Vec<Vec<u8>>,
    train_labels: Vec<u8>,
    test_data: Vec<Vec<u8>>,
    test_labels: Vec<u8>,
}

impl Mnist {
    pub fn new(mnist_path: &str) -> Mnist {
        // ------------------------------------ Get Training Data ---------------------------------
        info!("Reading MNIST training data.");
        let (magic_number, num_images, num_rows, num_cols, train_images) =
            parse_images(&[mnist_path, TRAIN_DATA_FILENAME].concat()).expect(
                &format!(
                    "Training data file \"{}{}\" not found; did you \
                     remember to download and extract it?",
                    mnist_path, TRAIN_DATA_FILENAME,
                )[..],
            );

        // Assert that numbers extracted from the file were as expected.
        assert_eq!(
            magic_number, IMAGES_MAGIC_NUMBER,
            "Magic number for training data does not match expected value."
        );
        assert_eq!(
            num_images, NUM_TRAIN_IMAGES,
            "Number of images in training data does not match expected value."
        );
        assert_eq!(
            num_rows, IMAGE_ROWS,
            "Number of rows per image in training data does not match expected value."
        );
        assert_eq!(
            num_cols, IMAGE_COLUMNS,
            "Numver of columns per image in training data does not match expected value."
        );

        // ------------------------------------ Get Testing Data ----------------------------------
        info!("Reading MNIST testing data.");
        let (magic_number, num_images, num_rows, num_cols, test_images) =
            parse_images(&[mnist_path, TEST_DATA_FILENAME].concat()).expect(
                &format!(
                    "Test data file \"{}{}\" not found; did you \
                     remember to download and extract it?",
                    mnist_path, TEST_DATA_FILENAME,
                )[..],
            );

        // Assert that numbers extracted from the file were as expected.
        assert_eq!(
            magic_number, IMAGES_MAGIC_NUMBER,
            "Magic number for testing data does not match expected value."
        );
        assert_eq!(
            num_images, NUM_TEST_IMAGES,
            "Number of images in testing data does not match expected value."
        );
        assert_eq!(
            num_rows, IMAGE_ROWS,
            "Number of rows per image in testing data does not match expected value."
        );
        assert_eq!(
            num_cols, IMAGE_COLUMNS,
            "Numver of columns per image in testing data does not match expected value."
        );

        // ---------------------------------- Get Training Labels ---------------------------------
        info!("Reading MNIST training labels.");
        let (magic_number, num_labels, train_labels) =
            parse_labels(&[mnist_path, TRAIN_LABEL_FILENAME].concat()).expect(
                &format!(
                    "Training label file \"{}{}\" not found; did you \
                     remember to download and extract it?",
                    mnist_path, TRAIN_LABEL_FILENAME,
                )[..],
            );

        // Assert that numbers extracted from the file were as expected.
        assert_eq!(
            magic_number, LABELS_MAGIC_NUMBER,
            "Magic number for training labels does not match expected value."
        );
        assert_eq!(
            num_labels, NUM_TRAIN_IMAGES,
            "Number of labels in training labels does not match expected value."
        );

        // ----------------------------------- Get Testing Data -----------------------------------
        info!("Reading MNIST testing labels.");
        let (magic_number, num_labels, test_labels) =
            parse_labels(&[mnist_path, TEST_LABEL_FILENAME].concat()).expect(
                &format!(
                    "Test labels file \"{}{}\" not found; did you \
                     remember to download and extract it?",
                    mnist_path, TEST_LABEL_FILENAME,
                )[..],
            );

        // Assert that numbers extracted from the file were as expected.
        assert_eq!(
            magic_number, LABELS_MAGIC_NUMBER,
            "Magic number for testing labels does not match expected value."
        );
        assert_eq!(
            num_labels, NUM_TEST_IMAGES,
            "Number of labels in testing labels does not match expected value."
        );

        Mnist {
            train_data: train_images,
            train_labels: train_labels,
            test_data: test_images,
            test_labels: test_labels,
        }
    }

    pub fn get_train_image(&self, index: usize) -> &Vec<u8> {
        &self.train_data[index]
    }

    pub fn get_train_label(&self, index: usize) -> u8 {
        self.train_labels[index]
    }

    pub fn get_test_image(&self, index: usize) -> &Vec<u8> {
        &self.test_data[index]
    }

    pub fn get_test_label(&self, index: usize) -> u8 {
        self.test_labels[index]
    }
}

pub fn print_sample_image(image: &Vec<u8>, label: u8) {
    // Check that the image isn't empty and has a valid number of rows.
    assert!(image.len() != 0, "There are no pixels in this image.");
    assert_eq!(
        image.len() % usize::try_from(IMAGE_ROWS).unwrap(),
        0,
        "Number of pixels does not evenly divide into number of rows."
    );

    println!("Sample image label: {} \nSample image:", label);

    // Print each row.
    for row in 0..IMAGE_ROWS {
        for col in 0..IMAGE_COLUMNS {
            if image[usize::try_from(row * IMAGE_COLUMNS + col).unwrap()] == 0 {
                print!("_");
            } else {
                print!("#");
            }
        }
        print!("\n");
    }
}

fn parse_images(filename: &str) -> io::Result<(u32, u32, u32, u32, Vec<Vec<u8>>)> {
    let images_data_bytes = fs::File::open(filename)?;
    let images_data_bytes = io::BufReader::new(images_data_bytes);
    let mut buffer_32: [u8; 4] = [0; 4];

    // Get the magic number.
    images_data_bytes
        .get_ref()
        .take(4)
        .read(&mut buffer_32)
        .unwrap();
    let magic_number = u32::from_be_bytes(buffer_32);

    // Get number of images.
    images_data_bytes
        .get_ref()
        .take(4)
        .read(&mut buffer_32)
        .unwrap();
    let num_images = u32::from_be_bytes(buffer_32);

    // Get number or rows per image.
    images_data_bytes
        .get_ref()
        .take(4)
        .read(&mut buffer_32)
        .unwrap();
    let num_rows = u32::from_be_bytes(buffer_32);

    // Get number or columns per image.
    images_data_bytes
        .get_ref()
        .take(4)
        .read(&mut buffer_32)
        .unwrap();
    let num_cols = u32::from_be_bytes(buffer_32);

    // Buffer for holding image pixels.
    let mut image_buffer: Vec<u8> = vec![0; usize::try_from(num_rows * num_cols).unwrap()];

    // Vector to hold all images in the file.
    let mut images: Vec<Vec<u8>> =
        Vec::with_capacity(usize::try_from(num_images * num_rows * num_cols).unwrap());

    // Get images from file.
    for _image in 0..num_images {
        images_data_bytes
            .get_ref()
            .take(u64::from(num_rows * num_cols))
            .read(&mut image_buffer)
            .unwrap();
        images.push(image_buffer.clone());
    }

    Ok((magic_number, num_images, num_rows, num_cols, images))
}

fn parse_labels(filename: &str) -> io::Result<(u32, u32, Vec<u8>)> {
    let labels_data_bytes = fs::File::open(filename)?;
    let labels_data_bytes = io::BufReader::new(labels_data_bytes);
    let mut buffer_32: [u8; 4] = [0; 4];

    // Get the magic number.
    labels_data_bytes
        .get_ref()
        .take(4)
        .read(&mut buffer_32)
        .unwrap();
    let magic_number = u32::from_be_bytes(buffer_32);

    // Get number of labels.
    labels_data_bytes
        .get_ref()
        .take(4)
        .read(&mut buffer_32)
        .unwrap();
    let num_labels = u32::from_be_bytes(buffer_32);

    // Buffer for holding image label.
    let mut label_buffer: [u8; 1] = [0; 1];

    // Vector to hold all labels in the file.
    let mut labels: Vec<u8> = Vec::with_capacity(usize::try_from(num_labels).unwrap());

    // Get labels from file.
    for _label in 0..num_labels {
        labels_data_bytes
            .get_ref()
            .take(1)
            .read(&mut label_buffer)
            .unwrap();
        labels.push(label_buffer[0]);
    }
    Ok((magic_number, num_labels, labels))
}
