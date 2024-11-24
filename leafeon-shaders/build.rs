use spirv_builder::{Capability, MetadataPrintout, SpirvBuilder};

const SHADERS: &[&str] = &["leafeon-mmul"];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for shader in SHADERS {
        SpirvBuilder::new(shader, "spirv-unknown-spv1.5")
            .capability(Capability::Linkage)
            .print_metadata(MetadataPrintout::Full)
            .build()?;
    }
    Ok(())
}
