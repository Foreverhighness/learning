use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use walkdir::WalkDir;
use xshell::{cmd, Shell};

use crate::clippy::CLIPPY_ARGS;

mod clippy;
mod flags;

static WORKSPACE_PATH: LazyLock<&Path> = LazyLock::new(|| {
    Box::leak(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace not found!")
            .to_path_buf()
            .into_boxed_path(),
    )
});

fn main() -> xshell::Result<()> {
    let cli = flags::Xtask::from_env_or_exit();

    let sh = &Shell::new()?;
    sh.change_dir(*WORKSPACE_PATH);

    match cli.subcommand {
        flags::XtaskCmd::Clean(_) => {
            for dir in WalkDir::new(*WORKSPACE_PATH)
                .max_depth(1)
                .into_iter()
                .filter_entry(|entry| entry.file_type().is_dir())
                .filter_map(Result::ok)
            {
                // make clean
                if dir.path().join("Makefile").exists() {
                    let path = dir.path();
                    cmd!(sh, "make clean --no-print-directory -C {path}").quiet().run()?;
                    continue;
                }
                // fallback to do nothing
            }

            // cargo clean at workspace
            cmd!(sh, "cargo clean").run()?;
        }
        flags::XtaskCmd::Clippy(_) => cmd!(sh, "cargo clippy -- {CLIPPY_ARGS...}").run()?,
    }

    Ok(())
}
