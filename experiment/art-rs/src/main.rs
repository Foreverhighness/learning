use std::io::{BufWriter, Write};

use art_rs::{PixelGenerator, MAX_COLOR, SCALE_FACTOR as SCALE_FACTOR_U32};
use clap::{arg, command, Parser};

const DEFAULT_WIDTH: u16 = 1024;
const DEFAULT_HEIGHT: u16 = 1024;
const DEFAULT_RADIUS: i16 = 1;
const DEFAULT_PROBABILITY: u32 = 100;
const SCALE_FACTOR: i64 = SCALE_FACTOR_U32 as _;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    /// The width of output picture
    #[arg(short = 'W', long, default_value_t = DEFAULT_WIDTH)]
    width: u16,

    /// The height of output picture
    #[arg(short = 'H', long, default_value_t = DEFAULT_HEIGHT)]
    height: u16,

    /// The radius of random walk
    #[arg(short, long, default_value_t = DEFAULT_RADIUS, value_parser = clap::value_parser!(i16).range(1..))]
    radius: i16,

    /// The probability to generate one pixel, scale by a factor of 100000, means 1000 -> 1%,
    /// default 100 -> 0.1%
    #[arg(short, long, default_value_t = DEFAULT_PROBABILITY, value_parser = clap::value_parser!(u32).range(1..=SCALE_FACTOR))]
    probability: u32,

    /// The seed of rand
    #[arg(short, long, default_value_t = 0)]
    seed: u64,

    /// Output picture path
    path: String,
}

fn main() -> std::io::Result<()> {
    let Cli {
        width,
        height,
        radius,
        probability,
        seed,
        path,
    } = Cli::parse();

    let (width, height) = (width as usize, height as usize);

    let mut gen = PixelGenerator::new(width, height, radius as isize, probability, seed);

    let file = std::fs::File::create(path)?;
    let mut file = BufWriter::new(file);

    // https://netpbm.sourceforge.net/doc/ppm.html
    file.write_all(format!("P6\n{width} {height}\n{MAX_COLOR}\n").as_bytes())?;
    for i in 0..height {
        for j in 0..width {
            file.write_all(&[gen.red(i, j), gen.green(i, j), gen.blue(i, j)])?;
        }
    }

    Ok(())
}
