use pest::error::LineColLocation;
use pest::Parser;
use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "ldv2.pest"]
pub struct LinkerScriptParser;

mod ast;

const RISCV: &str = "riscv.ld";
const LOONGARCH: &str = "loongarch.ld";
const ARM: &str = "arm.ld";
const ARM64: &str = "arm64.ld";
const X86: &str = "x86.ld";

fn main() {
    let filenames = [RISCV, LOONGARCH, ARM, ARM64, X86];
    for filename in filenames {
        let file =
            std::fs::read_to_string(filename).expect(&format!(r#"file "{filename}" not exist."#));

        let mut file = match LinkerScriptParser::parse(Rule::LinkerScript, &file) {
            Ok(file) => file,
            Err(e) => {
                let LineColLocation::Pos((line, column)) = e.line_col else {
                    unreachable!()
                };
                panic!(
                    "{e:?} \n{}/{filename}:{line}:{column}",
                    std::env::current_dir().unwrap().display()
                );
            }
        };

        let file = file.next().unwrap();
        let linker_script = ast::LinkerScript::parse(file);
        eprintln!("{}", linker_script.to_string());
    }
}
