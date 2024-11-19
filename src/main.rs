#![feature(random)]

use std::random::random;

use digit_recognition_rs::{model::Network, parser::load_data};
use indicatif::ProgressStyle;
use tracing::{instrument::WithSubscriber, Span};
use tracing_indicatif::IndicatifLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

fn main() -> anyhow::Result<()> {
    let indicatif_layer = IndicatifLayer::new().with_progress_style(
        ProgressStyle::default_bar(), //.template("{elapsed} {span_name} {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")?,
    );

    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_writer(indicatif_layer.get_stderr_writer()))
        .with(indicatif_layer)
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
    let network = network
        .train()
        .dataset(dataset)
        .accuracy(0.01)
        .epochs(1)
        .call();
    tracing::info!("finished training");
    Ok(())
}
