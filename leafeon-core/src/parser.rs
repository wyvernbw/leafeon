use std::{fs, path::PathBuf};

use anyhow::Context;
use leafeon_types::prelude::*;
use nom::{
    bytes::complete::take,
    combinator::{map, verify},
    multi::{count, many0},
    number::complete::be_u32,
    sequence::tuple,
    IResult,
};

#[bon::builder]
pub fn load_data(
    data_path: impl Into<PathBuf>,
    labels_path: impl Into<PathBuf>,
) -> anyhow::Result<Dataset> {
    let data = fs::read(data_path.into()).context("failed to read data file")?;
    let (_, (headers, images)) = parse_images(Box::leak(data.into_boxed_slice()))?;
    assert!(headers.image_count() == images.len() as u32);
    //Ok(dataset)
    let labels_data = fs::read(labels_path.into()).context("failed to read labels file")?;
    let (_, labels) = parse_labels(Box::leak(labels_data.into_boxed_slice()))?;
    let labeled_images = images.into_iter().zip(labels).collect::<Vec<_>>();
    Ok(Dataset {
        headers,
        images: labeled_images,
    })
}

fn parse_image_headers(input: &[u8]) -> IResult<&[u8], Headers> {
    let (rest, headers) = verify(count(be_u32, 4usize), |data: &[u32]| data.len() == 4)(input)?;

    let &[magic_number, number_of_images, rows, columns] = headers.as_slice() else {
        unreachable!();
    };

    Ok((
        rest,
        Headers {
            magic_number,
            image_count: number_of_images,
            rows,
            columns,
        },
    ))
}

fn parse_images(input: &[u8]) -> IResult<&[u8], (Headers, Vec<Image>)> {
    let (image_data, headers) = parse_image_headers(input)?;
    let byte_count = headers.rows as usize * headers.columns as usize;
    let (rest, images) = many0(map(map(take(byte_count), Vec::from), Image::new))(image_data)?;
    Ok((rest, (headers, images)))
}

fn parse_labels(input: &[u8]) -> IResult<&[u8], Vec<Label>> {
    let (labels, (_, count)) = tuple((be_u32, be_u32))(input)?;
    assert!(labels.len() == count as usize);
    Ok((labels, labels.to_vec()))
}
