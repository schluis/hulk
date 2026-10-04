//! Explicitly convert a legacy JSON5 parameter file; runtime loading stays literal.
use clap::Parser;
use color_eyre::Result;
use std::{fs, path::PathBuf};
use types::parameters::BallFilterParameters;

#[derive(Parser)]
#[command(
    about = "Convert legacy ball-filter zero-disable sentinels to explicit large limits; prints JSON to stdout"
)]
struct Args {
    /// A legacy file only: do not pass configurations already using literal limits.
    input: PathBuf,
}
fn main() -> Result<()> {
    let args = Args::parse();
    let mut value: serde_json::Value = json5::from_str(&fs::read_to_string(args.input)?)?;
    ball_filter_tuner::parameter_migration::from_legacy(&mut value)?;
    // Check that conversion did not leave a missing/malformed required field.
    let _: BallFilterParameters = serde_json::from_value(value.clone())?;
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}
