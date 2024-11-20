use std::path::Path;

use anyhow::Context;
use charming::{
    component::{Grid, VisualMap},
    datatype::{
        CompositeValue::Number,
        DataFrame, DataPoint,
        NumericValue::Float,
    },
    element::{Emphasis, ItemStyle, Label, Orient, Tooltip},
    series::{Heatmap, Series},
    Chart, HtmlRenderer,
};
use ndarray::Array2;

pub struct ImageLogger;

pub trait IntoHeatmapSeries {
    fn into_series(self, label: impl Into<String>) -> impl Into<Series>;
}

impl IntoHeatmapSeries for Array2<f32> {
    fn into_series(self, label: impl Into<String>) -> impl Into<Series> {
        let data: Vec<_> = self
            .rows()
            .into_iter()
            .map(|row| {
                let row = row
                    .iter()
                    .map(|x| {
                        //tracing::info!(?x);
                        //assert!(!x.is_nan());
                        DataPoint::Value(Number(Float(f64::from(*x))))
                    })
                    .collect::<Vec<_>>();
                DataFrame::from(row)
            })
            .collect();
        Heatmap::new()
            .name(label)
            .label(Label::new().show(true))
            .emphasis(
                Emphasis::new().item_style(
                    ItemStyle::new()
                        .shadow_blur(10)
                        .shadow_color("rgba(0, 0, 0, 0.5)"),
                ),
            )
            .data(data)
    }
}

impl ImageLogger {
    pub fn log(
        image: impl IntoHeatmapSeries,
        label: impl Into<String>,
        path: impl AsRef<Path>,
    ) -> anyhow::Result<()> {
        let label = label.into();
        let chart = Self::heatmap(image, label.clone())?;
        // Chart dimension 1000x800.
        let mut renderer = HtmlRenderer::new("my charts", 1000, 800);
        // Save the chart as HTML file.
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;
        renderer
            .save(&chart, path.join(label + ".html"))
            .context("Failed to save chart")?;
        Ok(())
    }
    pub fn log_many(
        images: Vec<impl IntoHeatmapSeries>,
        label: impl Into<String>,
        path: impl AsRef<Path>,
    ) -> anyhow::Result<()> {
        let label = label.into();

        let path = path.as_ref().join(label.clone());
        for (idx, image) in images.into_iter().enumerate() {
            Self::log(image, label.clone() + &format!("_{idx}"), &path)?;
        }
        Ok(())
    }
    fn heatmap(image: impl IntoHeatmapSeries, label: impl Into<String>) -> anyhow::Result<Chart> {
        let chart = Chart::new()
            .tooltip(Tooltip::new().position("top"))
            .grid(Grid::new().height("50%").top("10%"))
            .visual_map(
                VisualMap::new()
                    .min(-10.0)
                    .max(10.0)
                    .calculable(true)
                    .orient(Orient::Horizontal)
                    .left("center")
                    .bottom("15%"),
            )
            .series(image.into_series(label));
        Ok(chart)
    }
}
