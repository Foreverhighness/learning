use crate::UserC;

#[repr(C)]
pub struct ByteSliceView {
    pub ptr: *const u8,
    pub len: usize,
}

/// # Panics
///
/// panic if `comment` or `name` are not valid utf8 strings
#[expect(clippy::not_unsafe_ptr_arg_deref, reason = "for testing")]
#[unsafe(no_mangle)]
pub extern "C" fn Rust_write_comment_with_ByteSliceView(
    user: &mut UserC,
    comment: *const u8,
    comment_len: usize,
    name: ByteSliceView,
) {
    // SAFETY: caller ensure safety
    let comment = unsafe { std::slice::from_raw_parts(comment, comment_len) };
    let comment = std::str::from_utf8(comment).unwrap();

    // SAFETY: caller ensure safety
    let name = unsafe { std::slice::from_raw_parts(name.ptr, name.len) };
    let name = std::str::from_utf8(name).unwrap();
    println!("{name} {:x?} says: {comment}", user.uuid);

    user.comments_count += 1;
}
