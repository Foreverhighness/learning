use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    Command::new("gcc")
        .args(["src/thread_local.c", "-std=c17", "-O3", "-c", "-o"])
        .arg(format!("{out_dir}/thread_local.o"))
        .status()
        .unwrap();
    Command::new("ar")
        .args(["crus", "libthread_local.a", "thread_local.o"])
        .current_dir(Path::new(&out_dir))
        .status()
        .unwrap();
    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static=thread_local");
    println!("cargo:rerun-if-changed=src/thread_local.c");
}
