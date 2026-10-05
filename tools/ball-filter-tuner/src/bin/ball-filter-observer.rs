use ball_filter_tuner::observation::{Clip, ImageServer};
use clap::Parser;
use color_eyre::Result;
use std::{net::SocketAddr, path::PathBuf};
use types::parameters::BallFilterParameters;
#[derive(Parser)]
#[command(about = "Serve timestamped simulator replay images and filter state, without search")]
struct Args {
    #[arg(long,required=true,num_args=1..)]
    recordings: Vec<PathBuf>,
    /// Exact capture parameters, used to verify replay against recorded output.
    #[arg(long)]
    capture_parameters: PathBuf,
    /// Parameters to inspect. Defaults to the capture parameters.
    #[arg(long)]
    parameters: Option<PathBuf>,
    #[arg(long, default_value = "127.0.0.1:8765")]
    listen: SocketAddr,
}
fn main() -> Result<()> {
    color_eyre::install()?;
    let args = Args::parse();
    let read = |path: &PathBuf| -> Result<BallFilterParameters> {
        Ok(json5::from_str(&std::fs::read_to_string(path)?)?)
    };
    let capture = read(&args.capture_parameters)?;
    let parameters = args
        .parameters
        .as_ref()
        .map(read)
        .transpose()?
        .unwrap_or_else(|| capture.clone());
    let server = ImageServer::start(args.listen)?;
    for path in args.recordings {
        eprintln!("Loading {}", path.display());
        server.add(Clip::read(&path, &capture, &parameters)?);
    }
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(tokio::signal::ctrl_c())?;
    Ok(())
}
