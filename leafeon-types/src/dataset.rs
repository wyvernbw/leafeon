use ndarray::Array1;

#[derive(Debug, Clone)]

pub struct Image(Vec<u8>);

impl From<&Image> for Array1<f32> {
    fn from(Image(data): &Image) -> Self {
        let iterator = data.iter().map(|&x| x as f32 / u8::MAX as f32);
        Array1::from_iter(iterator)
    }
}

impl Image {
    pub fn new(data: Vec<u8>) -> Self {
        Self(data)
    }
}

pub type Label = u8;

#[derive(Debug, Clone)]
pub struct Dataset {
    pub headers: Headers,
    pub images: Vec<(Image, Label)>,
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

#[derive(Debug, Clone)]
pub struct Headers {
    pub magic_number: u32,
    pub image_count: u32,
    pub rows: u32,
    pub columns: u32,
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
