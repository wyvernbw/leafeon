#![feature(random)]

use std::random::random;

use anyhow::Context;
use clap::{Parser, Subcommand};
use digit_recognition_rs::{
    default_progress_style,
    model::{Activations, Network},
    parser::{load_data, Dataset},
};
use indicatif::ProgressStyle;
use inquire::{prompt_text, prompt_u32, Select, Text};
use strum::{Display, EnumIter, EnumString, IntoEnumIterator};
use tracing::{instrument::WithSubscriber, Span};
use tracing_indicatif::IndicatifLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, Layer};

#[derive(Debug, Parser)]
pub struct Cli {
    #[arg(short, default_value_t = false)]
    pub interactive: bool,
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Debug, Display, Subcommand, EnumIter)]
pub enum Command {
    Train { save_path: String },
    Run { weights: String, example: u32 },
    GetAccuracy { weights: String },
    GetTrainingAccuracy { weights: String },
}

pub fn untrained() -> Network {
    Network::untrained()
        .input_size(28 * 28)
        .layer_spec(&[128, 128, 32, 10])
        .call()
}

pub fn train(network: Network, dataset: Dataset) -> Network {
    network
        .train()
        .dataset(dataset)
        .accuracy(128.0 / 60_000.0)
        //.accuracy(1.0)
        .epochs(15)
        .learning_rate(0.0005)
        //.learning_rate(1.0)
        .call()
}

fn main() -> anyhow::Result<()> {
    init_tracing()?;
    let cli = Cli::parse();
    let command = cli.command.map_or_else(prompt_command, Ok)?;

    match command {
        Command::Train { save_path } => {
            let dataset = load_data()
                .labels_path("./data/train-labels-idx1-ubyte")
                .data_path("./data/train-images-idx3-ubyte")
                .call()?;
            let network = untrained();
            let network = train(network, dataset);

            tracing::info!("finished training");
            network.save_data(save_path)?;
            tracing::info!("succesfully saved train data");
        }
        Command::Run { weights, example } => {
            let network = Network::from_pretrained().path(weights).call()?;
            let dataset = load_data()
                .labels_path("./data/t10k-labels-idx1-ubyte")
                .data_path("./data/t10k-images-idx3-ubyte")
                .call()?;
            let (image, _) = dataset
                .images()
                .get(example as usize)
                .context("Invalid index!")?;
            dataset.print_image(example as usize);
            let example = Activations(image.into());
            let result = network
                .forward(example)
                .1
                .last()
                .context("No Output Layer!")?
                .0
                .clone();
            tracing::info!("{result}")
        }
        Command::GetAccuracy { weights } => {
            let dataset = load_data()
                .labels_path("./data/t10k-labels-idx1-ubyte")
                .data_path("./data/t10k-images-idx3-ubyte")
                .call()?;

            let network = untrained();
            tracing::info!(
                "untrained accuracy: {}%",
                network.accuracy(dataset.clone()) * 100.0
            );

            let network = Network::from_pretrained().path(weights).call()?;
            let accuracy = network.accuracy(dataset);
            let accuracy = accuracy * 100.0;
            tracing::info!("accuracy: {accuracy:3}%");
        }
        Command::GetTrainingAccuracy { weights } => {
            let dataset = load_data()
                .labels_path("./data/train-labels-idx1-ubyte")
                .data_path("./data/train-images-idx3-ubyte")
                .call()?;
            let network = untrained();
            tracing::info!(
                "untrained accuracy: {}%",
                network.accuracy(dataset.clone()) * 100.0
            );
            let network = Network::from_pretrained().path(weights).call()?;

            let accuracy = network.accuracy(dataset);
            let accuracy = accuracy * 100.0;
            tracing::info!("accuracy: {accuracy:3}%");
        }
    }

    Ok(())
}

fn init_tracing() -> anyhow::Result<()> {
    let indicatif_layer = IndicatifLayer::new().with_progress_style(default_progress_style());

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .without_time()
                .with_file(false)
                .with_writer(indicatif_layer.get_stderr_writer())
                .with_filter(tracing_subscriber::filter::LevelFilter::INFO),
        )
        .with(indicatif_layer)
        .init();
    Ok(())
}

fn prompt_command() -> anyhow::Result<Command> {
    let select = Select::new("", Command::iter().collect()).prompt()?;
    let command = match select {
        Command::Train { .. } => {
            let save_path = Text::new("Train data save path: ")
                .with_default("./train-data/data")
                .prompt()?;
            Command::Train { save_path }
        }
        Command::Run { .. } => {
            let weights = Text::new("Train data load path: ")
                .with_default("./train-data/data")
                .prompt()?;
            let example = prompt_u32("Which example to run?")?;
            Command::Run { weights, example }
        }
        Command::GetAccuracy { .. } => {
            let weights = Text::new("Train data load path: ")
                .with_default("./train-data/data")
                .prompt()?;
            Command::GetAccuracy { weights }
        }
        Command::GetTrainingAccuracy { .. } => {
            let weights = Text::new("Train data load path: ")
                .with_default("./train-data/data")
                .prompt()?;
            Command::GetTrainingAccuracy { weights }
        }
    };
    Ok(select)
}
