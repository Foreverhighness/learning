use crate::UserC;

#[repr(C)]
pub struct ByteSliceView {
    pub ptr: *const u8,
    pub len: usize,
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn Rust_write_comment_with_ByteSliceView(
    user: &mut UserC,
    comment: *const u8,
    comment_len: usize,
    name: ByteSliceView,
) {
    let comment = unsafe {
        let comment = std::slice::from_raw_parts(comment, comment_len);
        std::str::from_utf8_unchecked(comment)
    };

    let name = unsafe {
        let name = std::slice::from_raw_parts(name.ptr, name.len);
        std::str::from_utf8_unchecked(name)
    };
    println!("{name} {:x?} says: {comment}", user.uuid);

    user.comments_count += 1;
}
