#![feature(impl_trait_in_assoc_type)]
#![feature(generic_arg_infer)]
#![feature(iter_array_chunks)]
#![feature(iter_map_windows)]
#![feature(inherent_associated_types)]
#![feature(generic_const_exprs)]

use indicatif::ProgressStyle;

extern crate blas_src;

pub mod model;
pub mod parser;

pub fn default_progress_style() -> ProgressStyle {
    ProgressStyle::default_bar()
        .template(
            "{elapsed_precise} {span_name} {span_fields} {bar:40.cyan/pink} {pos:>7}/{len:7} {msg}",
        )
        .expect("Unable to create progress style")
}

pub fn default_progress_style_pink() -> ProgressStyle {
    ProgressStyle::default_bar()
        .template(
            "{elapsed_precise} {span_name} {span_fields} {bar:40.magenta} {pos:>7}/{len:7} {msg}",
        )
        .expect("Unable to create progress style")
}
