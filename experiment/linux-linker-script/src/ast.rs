#[derive(Debug)]
pub enum AstNode {
    LinkerScript(Vec<Command>),
    Command(Command),
    OutputFormat(OutputFormat),
    OutputArch(OutputArch),
    Entry(Entry),
    PHDRS(PHDRS),
    Sections(Sections),
    SymbolAssignment(SymbolAssignment),
    OutputSectionDescription(OutputSectionDescription),
    // Other variants can be added here for each rule in the grammar
}

#[derive(Debug)]
pub struct Command {
    pub kind: CommandKind,
}

#[derive(Debug)]
pub enum CommandKind {
    UnusedMacros,
    OutputFormat(OutputFormat),
    OutputArch(OutputArch),
    Entry(Entry),
    Phdrs(PHDRS),
    Sections(Sections),
    SymbolAssignment(SymbolAssignment),
}

#[derive(Debug)]
pub struct OutputFormat {
    pub formats: Vec<String>, // Assuming bfdname is a String
}

#[derive(Debug)]
pub struct OutputArch {
    pub architecture: String, // Assuming bfdarch is a String
}

#[derive(Debug)]
pub struct Entry {
    pub symbol: String,
}

#[derive(Debug)]
pub struct PHDRS {
    pub entries: Vec<PHDRSInner>,
}

#[derive(Debug)]
pub struct PHDRSInner {
    pub ident: String,
    pub flags: String,
    pub integer: i64, // Assuming integer is an i64
}

#[derive(Debug)]
pub struct Sections {
    pub commands: Vec<Command>,
}

#[derive(Debug)]
pub struct SymbolAssignment {
    pub symbol: String,
    pub operation: String, // "+" or "="
    pub expr: String,      // Expression as a string
}

#[derive(Debug)]
pub struct OutputSectionDescription {
    pub section: String,
    pub expr: Option<String>,
    pub ident: Option<String>,
    pub at_address: Option<String>,
    pub align: Option<String>,
    pub commands: Vec<OutputSectionCommand>,
    pub identifiers: Vec<String>,
    pub constants: Option<String>,
}

#[derive(Debug)]
pub enum OutputSectionCommand {
    UnusedMacros,
    SymbolAssignment(SymbolAssignment),
    OutputSectionData(OutputSectionData),
    InputSectionDescription(InputSectionDescription),
}

#[derive(Debug)]
pub struct OutputSectionData {
    pub byte: i64, // Assuming Byte is represented as an integer
}

#[derive(Debug)]
pub struct InputSectionDescription {
    pub input_section: String,           // Assuming input section is a string
    pub keep: Option<Vec<InputSection>>, // Keep sections if applicable
}

#[derive(Debug)]
pub struct InputSection {
    pub filename_wildcard: String,
    pub section_wildcards: Vec<SectionWildcard>,
}

#[derive(Debug)]
pub struct SectionWildcard {
    pub section: String,
    pub wildcard: Option<String>, // Optional "*"
}

// Implement methods as needed for AST manipulation
impl AstNode {
    pub fn new_linker_script(commands: Vec<Command>) -> Self {
        AstNode::LinkerScript(commands)
    }

    // Add other methods for interaction with the AST here
}

// Example method for `OutputFormat`
impl OutputFormat {
    pub fn new(formats: Vec<String>) -> Self {
        OutputFormat { formats }
    }
}

impl AstNode {
    pub fn to_string(&self) -> String {
        match self {
            AstNode::LinkerScript(commands) => {
                format!(
                    "LinkerScript {{\n{}\n}}",
                    commands
                        .iter()
                        .map(|cmd| cmd.to_string())
                        .collect::<Vec<String>>()
                        .join("\n")
                )
            }
            AstNode::Command(command) => command.to_string(),
            AstNode::OutputFormat(output_format) => output_format.to_string(),
            AstNode::OutputArch(output_arch) => output_arch.to_string(),
            AstNode::Entry(entry) => entry.to_string(),
            AstNode::PHDRS(phdrs) => phdrs.to_string(),
            AstNode::Sections(sections) => sections.to_string(),
            AstNode::SymbolAssignment(symbol_assignment) => symbol_assignment.to_string(),
            AstNode::OutputSectionDescription(osd) => osd.to_string(),
        }
    }
}

impl Command {
    pub fn to_string(&self) -> String {
        match &self.kind {
            CommandKind::UnusedMacros => "UnusedMacros".to_string(),
            CommandKind::OutputFormat(output_format) => output_format.to_string(),
            CommandKind::OutputArch(output_arch) => output_arch.to_string(),
            CommandKind::Entry(entry) => entry.to_string(),
            CommandKind::Phdrs(phdrs) => phdrs.to_string(),
            CommandKind::Sections(sections) => sections.to_string(),
            CommandKind::SymbolAssignment(symbol_assignment) => symbol_assignment.to_string(),
        }
    }
}

impl OutputFormat {
    pub fn to_string(&self) -> String {
        let formats_str = self.formats.join(", ");
        format!("OUTPUT_FORMAT({})", formats_str)
    }
}

impl OutputArch {
    pub fn to_string(&self) -> String {
        format!("OUTPUT_ARCH({})", self.architecture)
    }
}

impl Entry {
    pub fn to_string(&self) -> String {
        format!("ENTRY({})", self.symbol)
    }
}

impl PHDRS {
    pub fn to_string(&self) -> String {
        let entries_str = self
            .entries
            .iter()
            .map(|entry| entry.to_string())
            .collect::<Vec<String>>()
            .join("\n");
        format!("PHDRS {{\n{}\n}}", entries_str)
    }
}
impl PHDRSInner {
    pub fn to_string(&self) -> String {
        format!("{} {} FLAGS({})", self.ident, self.flags, self.integer)
    }
}

impl Sections {
    pub fn to_string(&self) -> String {
        let commands_str = self
            .commands
            .iter()
            .map(|cmd| cmd.to_string())
            .collect::<Vec<String>>()
            .join("\n");
        format!("SECTIONS {{\n{}\n}}", commands_str)
    }
}

impl SymbolAssignment {
    pub fn to_string(&self) -> String {
        format!("{} {} {}; ", self.symbol, self.operation, self.expr)
    }
}

impl OutputSectionDescription {
    pub fn to_string(&self) -> String {
        let commands_str = self
            .commands
            .iter()
            .map(|cmd| cmd.to_string())
            .collect::<Vec<String>>()
            .join("\n");
        let ident_str = self.identifiers.join(":");
        let constants_str = self.constants.clone().unwrap_or_default();

        format!(
            "{} {}: {} {{\n{commands_str}\n}} :{ident_str} ={constants_str}",
            self.section,
            self.expr.as_deref().unwrap_or(""),
            self.at_address.as_deref().unwrap_or(""),
        )
    }
}

impl OutputSectionCommand {
    pub fn to_string(&self) -> String {
        match self {
            OutputSectionCommand::UnusedMacros => "UnusedMacros".to_string(),
            OutputSectionCommand::SymbolAssignment(symbol_assignment) => {
                symbol_assignment.to_string()
            }
            OutputSectionCommand::OutputSectionData(data) => data.to_string(),
            OutputSectionCommand::InputSectionDescription(desc) => desc.to_string(),
        }
    }
}

impl OutputSectionData {
    pub fn to_string(&self) -> String {
        format!("BYTE({});", self.byte)
    }
}

impl InputSectionDescription {
    pub fn to_string(&self) -> String {
        if let Some(keep_sections) = &self.keep {
            let keep_str = keep_sections
                .iter()
                .map(|section| section.to_string())
                .collect::<Vec<String>>()
                .join(", ");
            format!("KEEP({})", keep_str)
        } else {
            self.input_section.clone()
        }
    }
}

impl InputSection {
    pub fn to_string(&self) -> String {
        let wildcards_str = self
            .section_wildcards
            .iter()
            .map(|wildcard| wildcard.to_string())
            .collect::<Vec<String>>()
            .join(", ");
        format!("{}({})", self.filename_wildcard, wildcards_str)
    }
}

impl SectionWildcard {
    pub fn to_string(&self) -> String {
        if let Some(wildcard) = &self.wildcard {
            format!("{}{}", self.section, wildcard)
        } else {
            self.section.clone()
        }
    }
}
