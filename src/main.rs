#![feature(random)]

use std::random::random;

use anyhow::Context;
use clap::{Parser, Subcommand};
use digit_recognition_rs::{
    default_progress_style,
    model::{Activations, Network},
    parser::load_data,
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
            let network = Network::untrained()
                .input_size(28 * 28)
                .layer_spec(&[128, 128, 32, 10])
                .call();
            let network = network
                .train()
                .dataset(dataset)
                .accuracy(0.01)
                .epochs(5)
                .learning_rate(0.001)
                .call();

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
            let example = Activations(image.into());
            tracing::info!("result: {:?}", network.forward(example).1.last())
        }
        Command::GetAccuracy { weights } => todo!(),
    }

    Ok(())
}

fn init_tracing() -> anyhow::Result<()> {
    let indicatif_layer = IndicatifLayer::new().with_progress_style(default_progress_style());

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .without_time()
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
    };
    Ok(select)
}
