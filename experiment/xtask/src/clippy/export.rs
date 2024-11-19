use crate::clippy::args::{ClippyLint, CLIPPY_LINTS};
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
        println!("cargo clippy --");
        for arg in CLIPPY_LINTS.iter().map(super::args::ClippyLint::compact_arg) {
            print!(" {arg}");
        }
        println!();
        Ok(None)
    }
}
impl Command for Cargo {
    fn cmd(self, _: &xshell::Shell) -> xshell::Result<Option<xshell::Cmd>> {
        println!("[lints.clippy]");
        for item in CLIPPY_LINTS.iter().map(ClippyLint::toml_item) {
            println!("{item}");
        }
        Ok(None)
    }
}
impl Command for Attr {
    fn cmd(self, _: &xshell::Shell) -> xshell::Result<Option<xshell::Cmd>> {
        for attr in CLIPPY_LINTS.iter().map(ClippyLint::attr) {
            println!("{attr}");
        }
        Ok(None)
    }
}
