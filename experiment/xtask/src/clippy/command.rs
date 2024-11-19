use xshell::cmd;

use super::args::CLIPPY_LINTS;
use crate::flags::{Clippy, Fix, Run};
use crate::Command;

impl Command for Clippy {
    fn cmd(self, sh: &xshell::Shell) -> xshell::Result<xshell::Cmd> {
        match self.subcommand {
            crate::flags::ClippyCmd::Run(run) => run.cmd(sh),
            crate::flags::ClippyCmd::Fix(fix) => fix.cmd(sh),
        }
    }
}

impl Command for Run {
    fn cmd(self, sh: &xshell::Shell) -> xshell::Result<xshell::Cmd> {
        let args = CLIPPY_LINTS.iter().map(|lint| lint.compact_arg());
        Ok(cmd!(sh, "cargo clippy -- {args...}"))
    }
}

impl Command for Fix {
    fn cmd(self, sh: &xshell::Shell) -> xshell::Result<xshell::Cmd> {
        // https://github.com/matklad/xshell/issues/34
        let force = self
            .force
            .then_some(&["--allow-no-vcs", "--allow-dirty", "--allow-staged"])
            .into_iter()
            .flatten();
        let args = CLIPPY_LINTS.iter().map(|lint| lint.compact_arg());
        Ok(cmd!(sh, "cargo clippy --fix {force...} -- {args...}"))
    }
}
