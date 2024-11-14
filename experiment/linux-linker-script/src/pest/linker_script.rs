use pest::iterators::Pair;

use super::ast::{AbstractSyntaxTreeNode, LinkerScriptIndentItem};
use super::commands::{Command, SectionsCommand};
use super::parser::Rule;

#[derive(Debug)]
pub struct LinkerScript {
    commands: Vec<Command>,
}

impl AbstractSyntaxTreeNode for LinkerScript {
    fn parse(pair: Pair<'_, Rule>) -> Self {
        assert!(matches!(pair.as_rule(), Rule::LinkerScript));
        let mut commands = Vec::new();
        for command in pair.into_inner() {
            if matches!(command.as_rule(), Rule::EOI) {
                break;
            }
            commands.push(Command::parse(command));
        }

        Self { commands }
    }
}

impl core::fmt::Display for LinkerScript {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let commands = self
            .commands
            .iter()
            .map(|cmd| cmd.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        write!(f, "{commands}",)
    }
}

#[derive(Debug)]
pub struct Sections {
    commands: Vec<SectionsCommand>,
}

impl AbstractSyntaxTreeNode for Sections {
    fn parse(pair: Pair<'_, Rule>) -> Self {
        assert!(matches!(pair.as_rule(), Rule::Sections));

        Self {
            commands: pair
                .into_inner()
                .into_iter()
                .map(SectionsCommand::parse)
                .collect(),
        }
    }
}

impl core::fmt::Display for Sections {
    /// SECTIONS {
    ///     sections-command
    ///     sections-command
    ///     …
    /// }
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let commands = self
            .commands
            .iter()
            .map(|cmd| cmd.display_with_indent())
            .collect::<Vec<_>>()
            .join("\n");
        write!(f, "SECTIONS {{\n{}\n}}", commands)
    }
}
