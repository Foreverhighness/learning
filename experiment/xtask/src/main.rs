use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use xshell::{Cmd, Shell};

mod clean;
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

trait Command {
    fn cmd(self, sh: &Shell) -> xshell::Result<Cmd>;
}

fn main() -> xshell::Result<()> {
    let cli = flags::Xtask::from_env_or_exit();

    let sh = &Shell::new()?;
    sh.change_dir(*WORKSPACE_PATH);

    let mut cmd = match cli.subcommand {
        flags::XtaskCmd::Clean(clean) => clean.cmd(sh)?,
        flags::XtaskCmd::Clippy(clippy) => clippy.cmd(sh)?,
    };

    if !cli.verbose {
        cmd = cmd.quiet();
    }

    cmd.run()
}
