use std::sync::Arc;

use anyhow::Context;
use axum::{
    body::Body,
    extract::State,
    routing::{get, post},
    Json, Router,
};
use base64::{prelude::BASE64_STANDARD, Engine};
use error::AppError;
use image::ImageFormat;
use leafeon_core::network::Network;
use leafeon_types::Activations;
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tower_http::{cors::CorsLayer, trace::TraceLayer};

pub mod error;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    if let Err(err) = tracing_subscriber::fmt::try_init() {
        tracing::warn!("Logger already initialized: {err}");
    };

    let network = Network::from_pretrained()
        .path("./train-data/weights.txt")
        .call()?;
    let app = Router::new()
        .route("/", get(root))
        .route("/predict", post(predict_image))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(Arc::new(network));
    let listener = TcpListener::bind("127.0.0.1:3000").await?;
    tracing::info!("Listening on http://{}", listener.local_addr()?);

    axum::serve(listener, app).await?;

    Ok(())
}

async fn root(State(network): State<Arc<Network>>) -> String {
    format!("leafeon is running! {:?}", network)
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Prediction {
    confidence: Vec<f32>,
    label: usize,
}

#[axum::debug_handler]
async fn predict_image(
    State(network): State<Arc<Network>>,
    body: String,
) -> Result<Json<Prediction>, AppError> {
    let body = body.split_once(",").context("Failed to split body")?.1;
    let body = BASE64_STANDARD
        .decode(body)
        .context("Failed to decode body")?;
    let image = image::load_from_memory_with_format(&body, ImageFormat::Png)?;
    let mut image = image.resize_exact(28, 28, image::imageops::FilterType::Triangle);

    image.invert();

    let values = image.into_luma8().into_vec();

    let network_values = network.forward(&Activations(
        (&leafeon_types::dataset::Image::new(values)).into(),
    ));
    let confidence = network_values
        .1
        .into_iter()
        .last()
        .context("No output layer")?;
    let label = Network::predict(confidence.clone()).0;
    let Activations(confidence) = confidence;
    let (confidence, _) = confidence.into_raw_vec_and_offset();

    Ok(Json(Prediction {
        confidence: confidence.to_vec(),
        label,
    }))
}
