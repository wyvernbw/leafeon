#![feature(generic_arg_infer)]
#![feature(iter_array_chunks)]
#![feature(iter_map_windows)]
#![feature(inherent_associated_types)]
#![feature(random)]
#![feature(generic_const_exprs)]

use std::random::random;

use model::{DModel, LayerSpec, OneNeuronExample};
use parser::load_data;
use tracing::Level;

pub mod model;
pub mod parser;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::fmt()
        .compact()
        .without_time()
        .with_max_level(Level::DEBUG)
        .init();
    let dataset = load_data()?;

    let test_idx = random::<u32>() % dataset.headers().image_count();
    let test_idx = test_idx as usize;
    tracing::info!("Loaded {} images", dataset.headers().image_count());
    dataset.print_image(test_idx);
    tracing::info!("Image {test_idx} is a {}", dataset.images()[test_idx].1);

    let model = DModel::untrained(&[
        (1, LayerSpec(28 * 28, 128)),
        (2, LayerSpec(128, 128)),
        (1, LayerSpec(128, 10)),
    ]);

    let model = model.train(dataset, 10);
    Ok(())
}
