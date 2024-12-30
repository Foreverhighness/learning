use crate::Command;
use crate::clippy::args::{CLIPPY_LINTS, ClippyLint};
use crate::flags::{Attr, Cargo, Cli, Export};

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
        let compact = self.compact;
        let long_arg = self.wide;
        let sep = if self.list { " \\\n    " } else { " " };
        print!("cargo clippy --");
        for arg in CLIPPY_LINTS.iter().map(|lint| lint.arg(compact, long_arg)) {
            print!("{sep}{arg}");
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
