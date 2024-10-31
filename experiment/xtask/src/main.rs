use std::fs;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use anyhow::Result;
use clap::{Parser, Subcommand};
use duct::cmd;
use walkdir::WalkDir;

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    action: Action,
}

#[derive(Debug, Subcommand)]
enum Action {
    /// Clean all
    Clean,
}

static WORKSPACE_PATH: LazyLock<&Path> = LazyLock::new(|| {
    Box::leak(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf()
            .into_boxed_path(),
    )
});

fn main() -> Result<()> {
    let Cli { action } = Cli::parse();

    match action {
        Action::Clean => {
            for dir in WalkDir::new(*WORKSPACE_PATH)
                .max_depth(1)
                .into_iter()
                .filter_entry(|entry| entry.file_type().is_dir())
                .filter_map(Result::ok)
            {
                // make clean
                if fs::exists(dir.path().join("Makefile"))? {
                    cmd!("make", "clean").dir(dir.path()).run()?;
                    continue;
                }
                // fallback to do nothing
            }

            // cargo clean at workspace
            cmd!("cargo", "clean").dir(*WORKSPACE_PATH).run()?;
        }
    }
    Ok(())
}
