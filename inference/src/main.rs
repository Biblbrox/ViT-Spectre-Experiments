#![allow(clippy::manual_retain)]

use std::{env, path::Path, time::Instant};

use image::{GenericImageView, imageops::FilterType};
use ndarray::{Array4, Axis};
use ort::{
    inputs,
    session::{Session, SessionOutputs},
    value::TensorRef,
};

const CIFAR100_MODEL_URL: &str = "../model.onnx";

#[rustfmt::skip]
pub const CIFAR100_CLASS_LABELS: [&str; 100] = [
"beaver", "dolphin", "otter", "seal", "whale",
"aquarium_fish", "flatfish", "ray", "shark", "trout",
"orchid", "poppy", "rose", "sunflower", "tulip",
"bottle", "bowl", "can", "cup", "plate",
"apple", "mushroom", "orange", "pear", "sweet_pepper",
"clock", "keyboard", "lamp", "telephone", "television",
"bed", "chair", "couch", "table", "wardrobe",
"bee", "beetle", "butterfly", "caterpillar", "cockroach",
"bear", "leopard", "lion", "tiger", "wolf",
"bridge", "castle", "house", "road", "skyscraper",
"cloud", "forest", "mountain", "plain", "sea",
"camel", "cattle", "chimpanzee", "elephant", "kangaroo",
"fox", "porcupine", "possum", "raccoon", "skunk",
"crab", "lobster", "snail", "spider", "worm",
"baby", "boy", "girl", "man", "woman",
"crocodile", "dinosaur", "lizard", "snake", "turtle",
"hamster", "mouse", "rabbit", "shrew", "squirrel",
"maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree",
"bicycle", "bus", "motorcycle", "pickup_truck", "train",
"lawn_mower", "rocket", "streetcar", "tank", "tractor",
];

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|x| x / sum).collect()
}

fn main() -> ort::Result<()> {
    // Load model
    let mut session = Session::builder()?.commit_from_file(CIFAR100_MODEL_URL)?;
    // Load & preprocess image
    let img = image::open(
        Path::new(env::current_exe().unwrap().to_str().unwrap())
            .join("data")
            .join("example.png"),
    )
    .expect("failed to load image")
    .resize_exact(32, 32, FilterType::CatmullRom)
    .to_rgb8();

    let mut input: Array4<f32> = Array4::zeros((1, 3, 32, 32));

    for (x, y, pixel) in img.enumerate_pixels() {
        let [r, g, b] = pixel.0;
        input[[0, 0, y as usize, x as usize]] = r as f32 / 255.0;
        input[[0, 1, y as usize, x as usize]] = g as f32 / 255.0;
        input[[0, 2, y as usize, x as usize]] = b as f32 / 255.0;
    }

    let mut i = 0;
    let now = Instant::now();
    const ITERATIONS: u32 = 100;
    while i < ITERATIONS {
        // Optional: CIFAR-100 mean/std normalization
        //let mean = [0.5071, 0.4867, 0.4408];
        //let std = [0.2675, 0.2565, 0.2761];
        //for c in 0..3 {
        //    input
        //        .index_axis_mut(Axis(1), c)
        //        .mapv_inplace(|x| (x - mean[c]) / std[c]);
        //}

        // Inference
        let outputs: SessionOutputs =
            session.run(inputs!["input" => TensorRef::from_array_view(&input)?])?;

        //let logits: Vec<f32> = outputs["output"]
        //    .try_extract_array::<f32>()?
        //    .index_axis(Axis(0), 0)
        //    .iter()
        //    .copied()
        //    .collect();

        //let probs = softmax(&logits);

        //// Top-1
        //let (class_id, confidence) = probs
        //    .iter()
        //    .enumerate()
        //    .max_by(|a, b| a.1.total_cmp(b.1))
        //    .unwrap();

        //println!(
        //    "Prediction: {} ({:.2}%)",
        //    CIFAR100_CLASS_LABELS[class_id],
        //    confidence * 100.0
        //);

        // Optional: Top-5
        //let mut top5: Vec<_> = probs.iter().enumerate().collect();
        //top5.sort_by(|a, b| b.1.total_cmp(a.1));

        //println!("Top-5:");
        //for (idx, prob) in top5.iter().take(5) {
        //    println!(
        //        "  {:>20}: {:.2}%",
        //        CIFAR100_CLASS_LABELS[*idx],
        //        *prob * 100.0
        //    );
        //}

        i += 1;
    }

    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed / ITERATIONS);

    Ok(())
}
