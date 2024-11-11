pub mod byte_slice_view;

#[repr(C)]
pub struct UserC {
    pub name: [u8; 32],
    pub comments_count: u64,
    pub uuid: [u8; 16],
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn Rust_write_comment(user: &mut UserC, comment: *const u8, comment_len: usize) {
    let comment = unsafe { std::slice::from_raw_parts(comment, comment_len) };
    let comment = unsafe { std::str::from_utf8_unchecked(comment) };
    println!("{:x?} says: {comment}", user.uuid);

    user.comments_count += 1;
}
