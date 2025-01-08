use core::ffi::c_int;

#[unsafe(export_name = "add")]
pub extern "C" fn add(left: c_int, right: c_int) -> c_int {
    println!("calling in rust");
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
