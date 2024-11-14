use data::load_data;

pub mod data {
    use std::{any, fs};

    use nom::{
        bytes::complete::take,
        combinator::{all_consuming, map, map_res, verify},
        multi::{count, many0},
        number::complete::be_u32,
        sequence::tuple,
        IResult,
    };

    const DATA_PATH: &str = "./data/train-images-idx3-ubyte";

    pub struct Image(Vec<u8>);

    pub fn load_data() -> anyhow::Result<(Headers, Vec<Image>)> {
        let data = fs::read(DATA_PATH)?;
        let (_, (headers, images)) = parse_images(Box::leak(data.into_boxed_slice()))?;
        Ok((headers, images))
    }

    pub struct Headers {
        pub magic_number: u32,
        pub number_of_images: u32,
        pub rows: u32,
        pub columns: u32,
    }

    fn parse_image_headers<'a>(input: &'a [u8]) -> IResult<&'a [u8], Headers> {
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
                number_of_images,
                rows,
                columns,
            },
        ))
    }

    fn parse_images<'a>(input: &'a [u8]) -> IResult<&'a [u8], (Headers, Vec<Image>)> {
        let (image_data, headers) = parse_image_headers(input)?;
        let byte_count = headers.rows as usize * headers.columns as usize;
        let (rest, images) = many0(map(map(take(byte_count), Vec::from), Image))(image_data)?;
        Ok((rest, (headers, images)))
    }
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let data = load_data()?;
    //tracing::info!("Loaded {} images", data.len());
    Ok(())
}
