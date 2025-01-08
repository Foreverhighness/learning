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
            if !dir.path().join("Makefile").exists() {
                // Makefile not exist, do nothing
                continue;
            }

            // make clean
            let path = dir.path();
            cmd!(sh, "make clean --no-print-directory -C {path}").quiet().run()?;
        }

        // cargo clean at workspace
        Ok(Some(cmd!(sh, "cargo clean")))
    }
}
