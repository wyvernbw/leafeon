#![feature(generic_const_exprs)]

use model::Model;
use parser::load_data;

pub mod model;
pub mod parser;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let dataset = load_data()?;
    let test_idx = 15345;
    tracing::info!("Loaded {} images", dataset.headers().image_count());
    dataset.print_image(test_idx);
    tracing::info!("Image {test_idx} is a {}", dataset.images()[test_idx].1);
    let alloc = Model::<28>::default();
    tracing::info!("somehow allocated enough memory");
    Ok(())
}
