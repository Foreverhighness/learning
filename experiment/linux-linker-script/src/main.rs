#[cfg(feature = "pest")]
mod pest;

#[cfg(feature = "pest")]
use pest::format;

const RISCV: &str = "riscv.ld";
const LOONGARCH: &str = "loongarch.ld";
const ARM: &str = "arm.ld";
const ARM64: &str = "arm64.ld";
const X86: &str = "x86.ld";

fn main() {
    let filenames = [RISCV, LOONGARCH, ARM, ARM64, X86];
    for filename in filenames {
        let linker_script = format(filename);
        std::fs::write(format!("format-{filename}"), linker_script).unwrap();
    }
}
