use pest::error::LineColLocation;
use pest::iterators::Pair;
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
        for command in file.into_inner() {
            print(command, 0);
        }
    }
}

fn print_with_indent(item: &str, indent: usize) {
    eprintln!("{:width$}{}", "", item, width = indent);
}

fn print(item: Pair<'_, Rule>, indent: usize) {
    match item.as_rule() {
        Rule::EOI => (),
        Rule::unused_marcos
        | Rule::OutputFormat
        | Rule::OutputArch
        | Rule::Entry
        | Rule::symbol_assignment => {
            print_with_indent(item.as_str(), indent);
        }
        Rule::PHDRS => {
            print_with_indent("PHDRS {", indent);
            for header in item.into_inner() {
                print_with_indent(header.as_str(), indent + 4);
            }
            print_with_indent("}", indent);
        }
        Rule::Sections => {
            print_with_indent("SECTIONS {", indent);
            for command in item.into_inner() {
                print(command, indent + 4);
            }
            print_with_indent("}", indent);
        }
        Rule::OutputSectionDescription => {}
        _ => panic!("rule {:?}", item.as_rule()),
    }
}

#[derive(Debug)]
enum AstNode {
    Marco,
    OutputFormat,
    OutputArch,
    Entry,
}

fn parse(data: &[u8]) -> Result<AstNode, String> {
    todo!()
}
