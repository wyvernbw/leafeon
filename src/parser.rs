use std::fs;

use nalgebra::{DVector, SVector};
use nom::{
    bytes::complete::take,
    combinator::{map, verify},
    multi::{count, many0},
    number::complete::be_u32,
    sequence::tuple,
    IResult,
};

const DATA_PATH: &str = "./data/train-images-idx3-ubyte";

#[derive(Debug, Clone)]

pub struct Image(Vec<u8>);

impl From<Image> for SVector<f32, { 28 * 28 }> {
    fn from(Image(data): Image) -> Self {
        let iterator = data.into_iter().map(|x| x as f32 / 255.0);
        SVector::from_iterator(iterator)
    }
}

impl From<&Image> for DVector<f32> {
    fn from(Image(data): &Image) -> Self {
        let iterator = data.into_iter().map(|&x| x as f32 / 255.0);
        DVector::from_iterator(28 * 28, iterator)
    }
}

pub type Label = u8;

#[derive(Debug, Clone)]
pub struct Dataset {
    headers: Headers,
    images: Vec<(Image, Label)>,
}

impl Dataset {
    pub fn print_image(&self, index: usize) {
        for y in 0..self.headers.rows {
            for x in 0..self.headers.columns {
                let pixel = self.images[index].0 .0
                    [y as usize * self.headers.columns as usize + x as usize];
                print!("{} ", if pixel > 0 { '#' } else { ' ' });
            }
            println!();
        }
    }

    pub fn headers(&self) -> &Headers {
        &self.headers
    }

    pub fn images(&self) -> &[(Image, Label)] {
        &self.images
    }
}

pub fn load_data() -> anyhow::Result<Dataset> {
    let data = fs::read(DATA_PATH)?;
    let (_, (headers, images)) = parse_images(Box::leak(data.into_boxed_slice()))?;
    assert!(headers.image_count == images.len() as u32);
    //Ok(dataset)
    let labels_data = fs::read("./data/train-labels-idx1-ubyte")?;
    let (_, labels) = parse_labels(Box::leak(labels_data.into_boxed_slice()))?;
    let labeled_images = images.into_iter().zip(labels).collect::<Vec<_>>();
    Ok(Dataset {
        headers,
        images: labeled_images,
    })
}

#[derive(Debug, Clone)]
pub struct Headers {
    magic_number: u32,
    image_count: u32,
    rows: u32,
    columns: u32,
}

impl Headers {
    pub fn magic_number(&self) -> u32 {
        self.magic_number
    }

    pub fn image_count(&self) -> u32 {
        self.image_count
    }

    pub fn rows(&self) -> u32 {
        self.rows
    }

    pub fn columns(&self) -> u32 {
        self.columns
    }
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
    let (rest, images) = many0(map(map(take(byte_count), Vec::from), Image))(image_data)?;
    Ok((rest, (headers, images)))
}

fn parse_labels(input: &[u8]) -> IResult<&[u8], Vec<Label>> {
    let (labels, (_, count)) = tuple((be_u32, be_u32))(input)?;
    assert!(labels.len() == count as usize);
    Ok((labels, labels.to_vec()))
}
