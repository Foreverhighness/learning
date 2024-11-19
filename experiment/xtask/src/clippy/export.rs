use crate::clippy::args::CLIPPY_LINTS;
use crate::flags::{Attr, Cargo, Cli, Export};
use crate::Command;

impl Command for Export {
    fn cmd(self, sh: &xshell::Shell) -> xshell::Result<Option<xshell::Cmd>> {
        match self.subcommand {
            crate::flags::ExportCmd::Cli(cli) => cli.cmd(sh),
            crate::flags::ExportCmd::Cargo(cargo) => cargo.cmd(sh),
            crate::flags::ExportCmd::Attr(attr) => attr.cmd(sh),
        }
    }
}

impl Command for Cli {
    fn cmd(self, _: &xshell::Shell) -> xshell::Result<Option<xshell::Cmd>> {
        let args = CLIPPY_LINTS
            .iter()
            .map(|lint| lint.compact_arg())
            .collect::<Vec<_>>()
            .join(" ");
        println!("cargo clippy -- {args}");
        Ok(None)
    }
}
impl Command for Cargo {
    fn cmd(self, _: &xshell::Shell) -> xshell::Result<Option<xshell::Cmd>> {
        todo!()
    }
}
impl Command for Attr {
    fn cmd(self, _: &xshell::Shell) -> xshell::Result<Option<xshell::Cmd>> {
        todo!()
    }
}
