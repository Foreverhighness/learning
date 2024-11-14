use derive_more::derive::Display;
use derive_more::From;
use pest::iterators::Pair;

use super::ast::{AbstractSyntaxTreeNode, LinkerScriptIndentItem};
use super::linker_script::Sections;
use super::parser::Rule;

#[derive(Debug, Display)]
pub enum Command {
    Macro(Macro),
    OutputFormat(OutputFormat),
    OutputArch(OutputArch),
    Entry(Entry),
    SymbolAssignment(SymbolAssignment),
    /// Need format item
    Assertion(Assertion),
    Phdrs(Phdrs),
    Sections(Sections),
}

impl AbstractSyntaxTreeNode for Command {
    fn parse(pair: Pair<'_, Rule>) -> Self {
        match pair.as_rule() {
            Rule::Macro => Command::Macro(Macro::parse(pair)),
            Rule::OutputFormat => Command::OutputFormat(OutputFormat::parse(pair)),
            Rule::OutputArch => Command::OutputArch(OutputArch::parse(pair)),
            Rule::Entry => Command::Entry(Entry::parse(pair)),
            Rule::SymbolAssignment => Command::SymbolAssignment(SymbolAssignment::parse(pair)),
            Rule::Assertion => Command::Assertion(Assertion::parse(pair)),
            Rule::Phdrs => Command::Phdrs(Phdrs::parse(pair)),
            Rule::Sections => Command::Sections(Sections::parse(pair)),
            _ => panic!("Unsupported rule: {:?}", pair.as_rule()),
        }
    }
}

#[derive(Debug, From, Display)]
pub struct Macro {
    inner: String,
}
#[derive(Debug, From, Display)]
pub struct OutputFormat {
    // binary_file_descriptor_name: String,
    inner: String,
}
#[derive(Debug, From, Display)]
pub struct OutputArch {
    // binary_file_descriptor_architecture: String,
    inner: String,
}
#[derive(Debug, From, Display)]
pub struct Entry {
    // entry_point: String,
    inner: String,
}
#[derive(Debug, From, Display)]
pub struct SymbolAssignment {
    inner: String,
    // symbol: String,
    // operation: String, // "+" or "="
    // expr: String,      // Expression as a string
}

#[derive(Debug)]
pub struct Assertion {
    // TODO(fh): replace its type with Expression
    expr: String,
    message: String,
}

impl AbstractSyntaxTreeNode for Assertion {
    fn parse(command: Pair<'_, Rule>) -> Self {
        assert!(matches!(command.as_rule(), Rule::Assertion));

        let mut assertion = command.into_inner();
        let expr = String::parse(assertion.next().unwrap());
        let message = String::parse(assertion.next().unwrap());

        Self { expr, message }
    }
}

impl core::fmt::Display for Assertion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ASSERT({}, {});", self.expr, self.message)
    }
}

#[derive(Debug)]
pub struct Phdrs {
    pub entries: Vec<Phdr>,
}

impl AbstractSyntaxTreeNode for Phdrs {
    fn parse(command: Pair<'_, Rule>) -> Self {
        assert!(matches!(command.as_rule(), Rule::Phdrs));
        Phdrs {
            entries: command.into_inner().into_iter().map(Phdr::parse).collect(),
        }
    }
}

impl core::fmt::Display for Phdrs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let entries_str = self
            .entries
            .iter()
            .map(|entry| entry.display_with_indent())
            .collect::<Vec<String>>()
            .join("\n");
        write!(f, "PHDRS {{\n{entries_str}\n}}")
    }
}

#[derive(Debug, From, Display)]
pub struct Phdr {
    inner: String,
    // pub name: String,
    // pub ty: String,
    // pub flags: Option<String>,
}

#[derive(Debug, Display)]
pub enum SectionsCommand {
    Macro(Macro),
    Entry(Entry),
    SymbolAssignment(SymbolAssignment),
    Assertion(Assertion),
    OutputSectionDescription(OutputSectionDescription),
}

impl AbstractSyntaxTreeNode for SectionsCommand {
    fn parse(pair: Pair<'_, Rule>) -> Self {
        assert!(matches!(
            pair.as_rule(),
            Rule::Macro
                | Rule::Entry
                | Rule::SymbolAssignment
                | Rule::Assertion
                | Rule::OutputSectionDescription
        ));

        match pair.as_rule() {
            Rule::Macro => SectionsCommand::Macro(Macro::parse(pair)),
            Rule::Entry => SectionsCommand::Entry(Entry::parse(pair)),
            Rule::SymbolAssignment => {
                SectionsCommand::SymbolAssignment(SymbolAssignment::parse(pair))
            }
            Rule::Assertion => SectionsCommand::Assertion(Assertion::parse(pair)),
            Rule::OutputSectionDescription => {
                SectionsCommand::OutputSectionDescription(OutputSectionDescription::parse(pair))
            }
            _ => unreachable!(),
        }
    }
}

#[derive(Debug)]
pub struct OutputSectionDescription {
    section: String,
    address: Option<String>,
    type_: Option<String>,
    at: Option<String>,
    align: Option<String>,
    commands: Vec<OutputSectionCommand>,
    phdrs: Vec<String>,
    fill_expr: Option<String>,
}

impl AbstractSyntaxTreeNode for OutputSectionDescription {
    fn parse(pair: Pair<'_, Rule>) -> Self {
        assert!(matches!(pair.as_rule(), Rule::OutputSectionDescription));

        let mut inner = pair.into_inner();

        let section = String::parse(inner.next().unwrap());
        let mut address = None;
        let mut type_ = None;
        let mut at = None;
        let mut align = None;
        let mut commands = Vec::new();
        let mut phdrs = Vec::new();
        let mut fill_expr = None;
        for pair in inner {
            match pair.as_rule() {
                Rule::expr => address = Some(String::parse(pair)),
                Rule::ident => type_ = Some(String::parse(pair)),
                Rule::AtAddress => at = Some(String::parse(pair)),
                Rule::align => align = Some(String::parse(pair)),
                Rule::Macro
                | Rule::SymbolAssignment
                | Rule::OutputSectionData
                | Rule::InputSectionDescription => commands.push(OutputSectionCommand::parse(pair)),
                Rule::OutputSectionPhdr => phdrs.push(String::parse(pair)),
                Rule::constant => fill_expr = Some(String::parse(pair)),
                _ => unreachable!(),
            }
        }

        Self {
            section,
            address,
            type_,
            at,
            align,
            commands,
            phdrs,
            fill_expr,
        }
    }
}

impl core::fmt::Display for OutputSectionDescription {
    /// section [address] [(type)] :
    /// [AT(lma)]
    /// [ALIGN(section_align) | ALIGN_WITH_INPUT]
    /// [SUBALIGN(subsection_align)]
    /// [constraint]
    /// {
    /// output-section-command
    /// output-section-command
    /// …
    /// } [>region] [AT>lma_region] [:phdr :phdr …] [=fillexp] [,]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let section = &self.section;
        write!(f, "{section}")?;
        if let Some(address) = self.address.as_ref() {
            write!(f, " {address}")?;
        }
        if let Some(type_) = self.type_.as_ref() {
            write!(f, " {type_}")?;
        }
        write!(f, " :")?;
        if let Some(at) = self.at.as_ref() {
            write!(f, " {at}")?;
        }
        if let Some(align) = self.align.as_ref() {
            write!(f, " {align}")?;
        }
        // Special judge length == 1
        if self.commands.len() == 1 {
            write!(f, " {{ {} }}", self.commands[0])?;
        } else {
            write!(f, " {{\n")?;
            for command in &self.commands {
                write!(f, "{}\n", command.display_with_indent())?;
            }
            write!(f, "}}")?;
        }
        if !self.phdrs.is_empty() {
            for phdr in &self.phdrs {
                write!(f, " {phdr}")?;
            }
        }
        if let Some(fill_expr) = self.fill_expr.as_ref() {
            write!(f, " {fill_expr}")?;
        }

        Ok(())
    }
}

#[derive(Debug, Display)]
pub enum OutputSectionCommand {
    Macro(Macro),
    SymbolAssignment(SymbolAssignment),
    OutputSectionData(OutputSectionData),
    InputSectionDescription(InputSectionDescription),
}

impl AbstractSyntaxTreeNode for OutputSectionCommand {
    fn parse(pair: Pair<'_, Rule>) -> Self {
        assert!(matches!(
            pair.as_rule(),
            Rule::Macro
                | Rule::SymbolAssignment
                | Rule::OutputSectionData
                | Rule::InputSectionDescription
        ));

        match pair.as_rule() {
            Rule::Macro => OutputSectionCommand::Macro(Macro::parse(pair)),
            Rule::SymbolAssignment => {
                OutputSectionCommand::SymbolAssignment(SymbolAssignment::parse(pair))
            }
            Rule::OutputSectionData => {
                OutputSectionCommand::OutputSectionData(OutputSectionData::parse(pair))
            }
            Rule::InputSectionDescription => {
                OutputSectionCommand::InputSectionDescription(InputSectionDescription::parse(pair))
            }
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, From, Display)]
pub struct OutputSectionData {
    inner: String,
}

#[derive(Debug, From, Display)]
pub struct InputSectionDescription {
    inner: String,
}
