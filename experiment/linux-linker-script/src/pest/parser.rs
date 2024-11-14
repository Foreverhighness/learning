use std::path::Path;

use pest::error::LineColLocation;
use pest::Parser;
use pest_derive::Parser;

use super::ast::AbstractSyntaxTreeNode;
use super::linker_script::LinkerScript;

#[derive(Parser)]
#[grammar = "ldv2.pest"]
pub struct LinkerScriptParser;

// TODO(fh): better signature design?
pub fn format(path: impl AsRef<Path>) -> String {
    let path = path.as_ref();
    let file = std::fs::read_to_string(path).unwrap_or_else(|_| panic!(r#"file "{path:?}" not exist."#));
    let mut file = match LinkerScriptParser::parse(Rule::LinkerScript, &file) {
        Ok(file) => file,
        Err(e) => {
            let LineColLocation::Pos((line, column)) = e.line_col else {
                unreachable!()
            };
            panic!(
                "{e:?} \n{}/{path:?}:{line}:{column}",
                std::env::current_dir().unwrap().display()
            );
        }
    };

    let file = file.next().unwrap();
    LinkerScript::parse(file).to_string()
}
