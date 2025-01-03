use walkdir::WalkDir;
use xshell::cmd;

use crate::flags::Clean;
use crate::{Command, WORKSPACE_PATH};

impl Command for Clean {
    fn cmd(self, sh: &xshell::Shell) -> xshell::Result<Option<xshell::Cmd>> {
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
        Ok(Some(cmd!(sh, "cargo clean")))
    }
}
