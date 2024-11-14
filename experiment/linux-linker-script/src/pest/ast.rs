use pest::iterators::Pair;

use super::parser::Rule;

pub trait AbstractSyntaxTreeNode: Sized {
    fn parse(pair: Pair<'_, Rule>) -> Self;
}

pub trait LinkerScriptIndentItem: AbstractSyntaxTreeNode {
    fn display_with_indent(&self) -> String;
}

/// Use custom display trait instead of `std::fmt::Display`?
impl<T: core::fmt::Display + AbstractSyntaxTreeNode> LinkerScriptIndentItem for T {
    fn display_with_indent(&self) -> String {
        let indent = " ".repeat(4);
        self.to_string()
            .lines()
            .map(|line| format!("{indent}{line}"))
            .collect::<Vec<String>>()
            .join("\n")
    }
}

// TODO(fh): remove impl, because `String` is not an ast node
impl<T: From<String>> AbstractSyntaxTreeNode for T {
    fn parse(pair: Pair<'_, Rule>) -> Self {
        pair.as_str().trim().to_owned().into()
    }
}
