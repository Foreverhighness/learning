#![feature(non_null_from_ref)]

mod behavior;

#[cfg(test)]
mod tests;

pub use behavior::basic::{run_when, CownPtr};

fn main() {
    println!("Hello, world!");
}
