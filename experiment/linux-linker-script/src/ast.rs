use pest::iterators::Pair;

use crate::Rule;

#[derive(Debug)]
pub struct LinkerScript {
    commands: Vec<Command>,
}

#[derive(Debug)]
pub enum Command {
    Macro,
    OutputFormat(OutputFormat),
    OutputArch(OutputArch),
    Entry(Entry),
    Phdrs(Phdrs),
    Sections(Sections),
    SymbolAssignment(SymbolAssignment),
}

#[derive(Debug)]
pub struct OutputFormat {
    pub binary_file_descriptor_name: String,
}

#[derive(Debug)]
pub struct OutputArch {
    pub binary_file_descriptor_architecture: String,
}

#[derive(Debug)]
pub struct Entry {
    pub entry_point: String,
}

#[derive(Debug)]
pub struct Phdrs {
    pub entries: Vec<Phdr>,
}

#[derive(Debug)]
pub struct Phdr {
    pub name: String,
    pub ty: String,
    pub flags: Option<String>,
}

#[derive(Debug)]
pub struct Sections {
    pub commands: Vec<SectionsCommand>,
}

#[derive(Debug)]
pub enum SectionsCommand {
    Macro,
    Entry(Entry),
    SymbolAssignment(SymbolAssignment),
    OutputSectionDescription(OutputSectionDescription),
}

#[derive(Debug)]
pub struct SymbolAssignment {
    pub inner: String,
    // pub symbol: String,
    // pub operation: String, // "+" or "="
    // pub expr: String,      // Expression as a string
}

#[derive(Debug)]
pub struct OutputSectionDescription {
    pub section: String,
    pub address: Option<String>,
    pub ty: Option<String>,
    pub at: Option<String>,
    pub align: Option<String>,
    pub commands: Vec<OutputSectionCommand>,
    pub phdrs: Vec<String>,
    pub fill_expr: Option<String>,
}

#[derive(Debug)]
pub enum OutputSectionCommand {
    Macro,
    SymbolAssignment(SymbolAssignment),
    OutputSectionData(OutputSectionData),
    InputSectionDescription(InputSectionDescription),
}

#[derive(Debug)]
pub struct OutputSectionData {
    pub inner: String,
}

#[derive(Debug)]
pub struct InputSectionDescription {
    pub input_section: String, // Assuming input section is a string
}

#[derive(Debug)]
pub struct Expression {
    // TODO(fh): replace with (lhs, op, rhs)
    expr: String,
}

// Example method for `OutputFormat`
impl OutputFormat {
    pub fn new(bfdname: String) -> Self {
        OutputFormat {
            binary_file_descriptor_name: bfdname,
        }
    }

    fn parse(command: Pair<'_, Rule>) -> OutputFormat {
        match command.as_rule() {
            Rule::OutputFormat => {
                todo!()
            }
            _ => unreachable!(),
        }
    }
}

impl Command {
    pub fn to_string_with_indent(&self, indent: usize) -> String {
        match &self {
            Command::Macro => todo!(),
            Command::OutputFormat(output_format) => output_format.to_string_with_indent(0),
            Command::OutputArch(output_arch) => output_arch.to_string_with_indent(0),
            Command::Entry(entry) => entry.to_string_with_indent(0),
            Command::Phdrs(phdrs) => phdrs.to_string_with_indent(0),
            Command::Sections(sections) => sections.to_string_with_indent(0),
            Command::SymbolAssignment(symbol_assignment) => {
                symbol_assignment.to_string_with_indent(0)
            }
        }
    }

    fn parse(command: Pair<'_, Rule>) -> Self {
        match command.as_rule() {
            Rule::unused_marcos => Command::Macro,
            Rule::OutputFormat => Command::OutputFormat(OutputFormat::parse(command)),
            Rule::OutputArch => Command::OutputArch(OutputArch::parse(command)),
            Rule::Entry => Command::Entry(Entry::parse(command)),
            Rule::PHDRS => Command::Phdrs(Phdrs::parse(command)),
            Rule::Sections => Command::Sections(Sections::parse(command)),
            Rule::symbol_assignment => Command::SymbolAssignment(SymbolAssignment::parse(command)),
            _ => panic!("Unsupported rule: {:?}", command.as_rule()),
        }
    }
}

impl OutputFormat {
    pub fn to_string_with_indent(&self, indent: usize) -> String {
        format!("OUTPUT_FORMAT({})", self.binary_file_descriptor_name)
    }
}

impl OutputArch {
    pub fn to_string_with_indent(&self, indent: usize) -> String {
        format!("OUTPUT_ARCH({})", self.binary_file_descriptor_architecture)
    }

    fn parse(command: Pair<'_, Rule>) -> OutputArch {
        assert!(matches!(command.as_rule(), Rule::OutputArch));
        OutputArch {
            binary_file_descriptor_architecture: command.into_inner().as_str().to_owned(),
        }
    }
}

impl Entry {
    pub fn to_string_with_indent(&self, indent: usize) -> String {
        format!("ENTRY({})", self.entry_point)
    }

    fn parse(command: Pair<'_, Rule>) -> Entry {
        assert!(matches!(command.as_rule(), Rule::Entry));
        Entry {
            entry_point: command.into_inner().as_str().to_owned(),
        }
    }
}

impl Phdrs {
    pub fn to_string_with_indent(&self, indent: usize) -> String {
        let entries_str = self
            .entries
            .iter()
            .map(|entry| entry.to_string_with_indent(indent))
            .collect::<Vec<String>>()
            .join("\n");
        format!("PHDRS {{\n{entries_str}\n}}")
    }

    fn parse(command: Pair<'_, Rule>) -> Phdrs {
        assert!(matches!(command.as_rule(), Rule::PHDRS));
        Phdrs {
            entries: command.into_inner().into_iter().map(Phdr::parse).collect(),
        }
    }
}
impl Phdr {
    pub fn to_string_with_indent(&self, indent: usize) -> String {
        let flags = self
            .flags
            .as_ref()
            .map_or(String::new(), |flags| format!("FLAGS({flags}"));
        format!("{} {} {flags};", self.name, self.ty)
    }

    fn parse(command: Pair<'_, Rule>) -> Phdr {
        assert!(matches!(command.as_rule(), Rule::PHDRS_inner));
        let phdr = command.into_inner();

        let name = phdr.find_first_tagged("name").unwrap().as_str().to_owned();
        let ty = phdr.find_first_tagged("ty").unwrap().as_str().to_owned();
        let flags = phdr
            .find_first_tagged("flag")
            .map(|pair| pair.as_str().to_owned());
        let t = Phdr { name, ty, flags };
        eprintln!("{}", t.to_string_with_indent(0));
        t
    }
}

impl Sections {
    pub fn to_string_with_indent(&self, indent: usize) -> String {
        let commands_str = self
            .commands
            .iter()
            .map(|cmd| cmd.to_string_with_indent(indent))
            .collect::<Vec<String>>()
            .join("\n");
        format!("SECTIONS {{\n{}\n}}", commands_str)
    }

    fn parse(command: Pair<'_, Rule>) -> Sections {
        assert!(matches!(command.as_rule(), Rule::Sections));

        Self {
            commands: command
                .into_inner()
                .into_iter()
                .map(SectionsCommand::parse)
                .collect(),
        }
    }
}

impl SectionsCommand {
    pub fn to_string_with_indent(&self, indent: usize) -> String {
        match self {
            SectionsCommand::Macro => todo!(),
            SectionsCommand::Entry(entry) => entry.to_string_with_indent(indent),
            SectionsCommand::SymbolAssignment(symbol_assignment) => {
                symbol_assignment.to_string_with_indent(indent)
            }
            SectionsCommand::OutputSectionDescription(output_section_description) => {
                output_section_description.to_string_with_indent(indent)
            }
        }
    }

    fn parse(command: Pair<'_, Rule>) -> Self {
        assert!(matches!(
            command.as_rule(),
            Rule::unused_marcos
                | Rule::Entry
                | Rule::symbol_assignment
                | Rule::OutputSectionDescription
        ));

        match command.as_rule() {
            Rule::unused_marcos => SectionsCommand::Macro,
            Rule::Entry => SectionsCommand::Entry(Entry::parse(command)),
            Rule::symbol_assignment => {
                SectionsCommand::SymbolAssignment(SymbolAssignment::parse(command))
            }
            Rule::OutputSectionDescription => {
                SectionsCommand::OutputSectionDescription(OutputSectionDescription::parse(command))
            }
            _ => unreachable!(),
        }
    }
}

impl SymbolAssignment {
    pub fn to_string_with_indent(&self, indent: usize) -> String {
        format!("{indent}{}", self.inner, indent = " ".repeat(indent))
    }

    fn parse(command: Pair<'_, Rule>) -> SymbolAssignment {
        assert!(matches!(command.as_rule(), Rule::symbol_assignment));

        Self {
            inner: command.as_str().to_owned(),
        }
    }
}

impl OutputSectionDescription {
    pub fn to_string_with_indent(&self, indent: usize) -> String {
        let commands_str = self
            .commands
            .iter()
            .map(|cmd| cmd.to_string_with_indent(indent))
            .collect::<Vec<String>>()
            .join("\n");
        let ident_str = self.phdrs.join(":");
        let constants_str = self.fill_expr.clone().unwrap_or_default();

        format!(
            "{} {}: {} {{\n{commands_str}\n}} :{ident_str} ={constants_str}",
            self.section,
            self.address.as_deref().unwrap_or(""),
            self.at.as_deref().unwrap_or(""),
        )
    }

    fn parse(command: Pair<'_, Rule>) -> Self {
        assert!(matches!(command.as_rule(), Rule::OutputSectionDescription));

        let inner = command.into_inner();
        let section = inner
            .find_first_tagged("section")
            .unwrap()
            .as_str()
            .to_owned();
        let address = inner
            .find_first_tagged("address")
            .map(|x| x.as_str().to_owned());
        let ty = inner.find_first_tagged("ty").map(|x| x.as_str().to_owned());
        let at = inner.find_first_tagged("at").map(|x| x.as_str().to_owned());
        let align = inner
            .find_first_tagged("align")
            .map(|x| x.as_str().to_owned());
        let commands = inner
            .clone()
            .find_tagged("commands")
            .map(OutputSectionCommand::parse)
            .collect();
        let phdrs = inner
            .clone()
            .find_tagged("phdrs")
            .map(|x| x.as_str().to_owned())
            .collect();
        let fill_expr = inner
            .find_first_tagged("fillexp")
            .map(|x| x.as_str().to_owned());

        Self {
            section,
            address,
            ty,
            at,
            align,
            commands,
            phdrs,
            fill_expr,
        }
    }
}

impl OutputSectionCommand {
    pub fn to_string_with_indent(&self, indent: usize) -> String {
        match self {
            OutputSectionCommand::Macro => todo!(),
            OutputSectionCommand::SymbolAssignment(symbol_assignment) => {
                symbol_assignment.to_string_with_indent(indent)
            }
            OutputSectionCommand::OutputSectionData(data) => data.to_string_with_indent(indent),
            OutputSectionCommand::InputSectionDescription(desc) => {
                desc.to_string_with_indent(indent)
            }
        }
    }

    fn parse(command: Pair<'_, Rule>) -> Self {
        assert!(matches!(
            command.as_rule(),
            Rule::unused_marcos
                | Rule::symbol_assignment
                | Rule::OutputSectionData
                | Rule::InputSectionDescription
        ));

        match command.as_rule() {
            Rule::unused_marcos => OutputSectionCommand::Macro,
            Rule::symbol_assignment => {
                OutputSectionCommand::SymbolAssignment(SymbolAssignment::parse(command))
            }
            Rule::OutputSectionData => {
                OutputSectionCommand::OutputSectionData(OutputSectionData::parse(command))
            }
            Rule::InputSectionDescription => OutputSectionCommand::InputSectionDescription(
                InputSectionDescription::parse(command),
            ),
            _ => unreachable!(),
        }
    }
}

impl OutputSectionData {
    pub fn to_string_with_indent(&self, indent: usize) -> String {
        format!("{}", self.inner)
    }

    fn parse(command: Pair<'_, Rule>) -> Self {
        assert!(matches!(command.as_rule(), Rule::OutputSectionData));

        Self {
            inner: command.as_str().to_owned(),
        }
    }
}

impl InputSectionDescription {
    pub fn to_string_with_indent(&self, indent: usize) -> String {
        format!(
            "{indent}{}",
            self.input_section,
            indent = " ".repeat(indent)
        )
    }

    fn parse(command: Pair<'_, Rule>) -> Self {
        assert!(matches!(command.as_rule(), Rule::InputSectionDescription));

        Self {
            input_section: command.as_str().to_owned(),
        }
    }
}

impl LinkerScript {
    pub fn parse(file: Pair<'_, Rule>) -> Self {
        assert!(matches!(file.as_rule(), Rule::LinkerScript));
        let mut commands = Vec::new();
        for command in file.into_inner() {
            if matches!(command.as_rule(), Rule::EOI) {
                break;
            }
            commands.push(Command::parse(command));
        }

        Self { commands }
    }

    pub fn to_string(&self) -> String {
        self.commands
            .iter()
            .map(|cmd| cmd.to_string_with_indent(0))
            .collect::<Vec<_>>()
            .join("\n")
    }
}
