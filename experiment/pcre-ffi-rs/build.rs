use cmake::Config;
fn main() {
    let dst = Config::new("pcre2").build_target("pcre2-8-static").build();
    println!(
        "cargo:rustc-link-search=native={}",
        dst.join("build").display()
    );
    println!("cargo:rustc-link-lib=static=pcre2-8");
    println!("cargo:rerun-if-changed=build.rs");
}
