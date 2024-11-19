#![feature(impl_trait_in_assoc_type)]
#![feature(generic_arg_infer)]
#![feature(iter_array_chunks)]
#![feature(iter_map_windows)]
#![feature(inherent_associated_types)]
#![feature(random)]
#![feature(generic_const_exprs)]

use std::{os::unix::net, random::random};

use digit_recognition_rs::{model::Network, parser::load_data};
use tracing::Level;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::fmt()
        .compact()
        .without_time()
        //.with_max_level(Level::DEBUG)
        .init();
    let dataset = load_data()?;

    let test_idx = random::<u32>() % dataset.headers().image_count();
    let test_idx = test_idx as usize;
    tracing::info!("Loaded {} images", dataset.headers().image_count());
    dataset.print_image(test_idx);
    tracing::info!("Image {test_idx} is a {}", dataset.images()[test_idx].1);

    let network = Network::untrained()
        .input_size(28 * 28)
        .layer_spec(&[128, 128, 32, 10])
        .call();
    tracing::info!(?network);
    let network = network
        .train()
        .dataset(dataset)
        .accuracy(0.01)
        .epochs(1)
        .call();
    tracing::info!("finished training");
    Ok(())
}
