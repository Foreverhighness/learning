pub mod byte_slice_view;

#[repr(C)]
pub struct UserC {
    pub name: [u8; 32],
    pub comments_count: u64,
    pub uuid: [u8; 16],
}

/// # Panics
///
/// panic if `comment` is not a valid utf8 string
#[expect(clippy::not_unsafe_ptr_arg_deref, reason = "for testing")]
#[no_mangle]
pub extern "C" fn Rust_write_comment(user: &mut UserC, comment: *const u8, comment_len: usize) {
    // SAFETY: Caller guarantee safety
    let comment = unsafe { std::slice::from_raw_parts(comment, comment_len) };
    let comment = std::str::from_utf8(comment).unwrap();
    println!("{:x?} says: {comment}", user.uuid);

    user.comments_count += 1;
}
