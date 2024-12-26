use xshell::{Cmd, cmd};

use super::args::{CLIPPY_LINTS, ClippyLint};
use crate::Command;
use crate::flags::{Clippy, Fix, Run};

impl Command for Clippy {
    fn cmd(self, sh: &xshell::Shell) -> xshell::Result<Option<Cmd>> {
        match self.subcommand {
            crate::flags::ClippyCmd::Run(run) => run.cmd(sh),
            crate::flags::ClippyCmd::Fix(fix) => fix.cmd(sh),
            crate::flags::ClippyCmd::Export(export) => export.cmd(sh),
        }
    }
}

impl Command for Run {
    fn cmd(self, sh: &xshell::Shell) -> xshell::Result<Option<Cmd>> {
        let args = CLIPPY_LINTS.iter().map(ClippyLint::compact_arg);
        Ok(Some(cmd!(sh, "cargo clippy -- {args...}")))
    }
}

impl Command for Fix {
    fn cmd(self, sh: &xshell::Shell) -> xshell::Result<Option<xshell::Cmd>> {
        // https://github.com/matklad/xshell/issues/34
        let force = self
            .force
            .then_some(&["--allow-no-vcs", "--allow-dirty", "--allow-staged"])
            .into_iter()
            .flatten();
        let args = CLIPPY_LINTS.iter().map(ClippyLint::compact_arg);
        Ok(Some(cmd!(sh, "cargo clippy --fix {force...} -- {args...}")))
    }
}
