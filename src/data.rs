use std::fs;

use nom::{
    bytes::complete::take,
    combinator::{map, map_res, verify},
    multi::{count, many0},
    IResult,
};

const DATA_PATH: &str = "./data/train-images-idx3-ubyte";

#[derive(Debug, Clone)]

pub struct Image(Vec<u8>);

#[derive(Debug, Clone)]
pub struct Dataset {
    headers: Headers,
    images: Vec<Image>,
}

impl Dataset {
    pub fn print_image(&self, index: usize) {
        for y in 0..self.headers.rows {
            for x in 0..self.headers.columns {
                let pixel =
                    self.images[index].0[y as usize * self.headers.columns as usize + x as usize];
                print!("{} ", if pixel > 0 { '#' } else { ' ' });
            }
            println!();
        }
    }

    pub fn headers(&self) -> &Headers {
        &self.headers
    }

    pub fn images(&self) -> &[Image] {
        &self.images
    }
}

pub fn load_data() -> anyhow::Result<Dataset> {
    let data = fs::read(DATA_PATH)?;
    let (_, dataset) = parse_images(Box::leak(data.into_boxed_slice()))?;
    assert!(dataset.headers.image_count == dataset.images.len() as u32);
    Ok(dataset)
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
    fn slice_to_u32(slice: &[u8]) -> anyhow::Result<u32> {
        Ok(u32::from_be_bytes(slice.try_into()?))
    }
    let (rest, headers) = verify(
        count(map_res(take(4usize), slice_to_u32), 4usize),
        |data: &[u32]| data.len() == 4,
    )(input)?;

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

fn parse_images(input: &[u8]) -> IResult<&[u8], Dataset> {
    let (image_data, headers) = parse_image_headers(input)?;
    let byte_count = headers.rows as usize * headers.columns as usize;
    let (rest, images) = many0(map(map(take(byte_count), Vec::from), Image))(image_data)?;
    Ok((rest, Dataset { headers, images }))
}
