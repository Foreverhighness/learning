use std::os::raw::c_int;

#[link(name="cachelab")]
extern "C" {
    fn printSummary(hits: c_int, misses: c_int, evictions: c_int);
}

fn main() {
    unsafe {
        printSummary(0, 0, 0);
    }
}