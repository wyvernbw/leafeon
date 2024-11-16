#![feature(iter_array_chunks)]
#![feature(iter_map_windows)]
#![feature(inherent_associated_types)]
#![feature(random)]
#![feature(generic_const_exprs)]

use std::{random::random, thread, time::Duration};

use model::OneNeuronExample;
use parser::load_data;
use tracing::Level;

pub mod model;
pub mod parser;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::fmt()
        .pretty()
        .with_max_level(Level::DEBUG)
        .init();
    let dataset = load_data()?;

    let test_idx = random::<u32>() % dataset.headers().image_count();
    let test_idx = test_idx as usize;
    tracing::info!("Loaded {} images", dataset.headers().image_count());
    dataset.print_image(test_idx);
    tracing::info!("Image {test_idx} is a {}", dataset.images()[test_idx].1);

    let example = OneNeuronExample::random(3);
    tracing::debug!("{example:#?}");
    tracing::info!("example.predict(0.5) = {}", example.predict(0.5));
    let example = example.backpropagate(1.0, dataset, 100);
    tracing::info!("after training: {example:#?}");
    tracing::info!(example = ?example.predict(0.5));
    Ok(())
}
