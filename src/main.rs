use data::load_data;

pub mod data;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let dataset = load_data()?;
    tracing::info!("Loaded {} images", dataset.headers().image_count());
    dataset.print_image(1);
    Ok(())
}
