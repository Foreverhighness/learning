#![allow(non_camel_case_types, reason = "FFI")]
#![allow(non_snake_case, reason = "FFI")]
use std::ffi::{c_void, CString};
use std::io::BufRead;
use std::os::raw::{c_int, c_uchar};

type pcre2_code = c_void;
type pcre2_match_data = c_void;
type pcre2_match_context = c_void;
type pcre2_general_context = c_void;

type PCRE2_SPTR = *const c_uchar;
type PCRE2_SIZE = usize;

#[link(name = "pcre2-8")]
extern "C" {
    fn pcre2_compile_8(
        pattern: PCRE2_SPTR,
        length: PCRE2_SIZE,
        options: u32,
        errorcode: *mut c_int,
        erroroffset: *mut PCRE2_SIZE,
        pcre_compile_context: *mut c_void,
    ) -> *mut pcre2_code;

    fn pcre2_match_data_create_from_pattern_8(
        code: *const pcre2_code,
        gcontext: *mut pcre2_general_context,
    ) -> *mut pcre2_match_data;

    fn pcre2_match_8(
        code: *const pcre2_code,
        subject: PCRE2_SPTR,
        length: PCRE2_SIZE,
        startoffset: PCRE2_SIZE,
        options: u32,
        match_data: *mut pcre2_match_data,
        mcontext: *mut pcre2_match_context,
    ) -> c_int;

    fn pcre2_get_ovector_pointer_8(match_data: *mut pcre2_match_data) -> *mut PCRE2_SIZE;
    fn pcre2_match_data_free_8(match_data: *mut pcre2_match_data);
    fn pcre2_code_free_8(code: *mut pcre2_code);
}

fn send_result(result: String) {
    use std::net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket};
    static SOCKET: std::sync::LazyLock<UdpSocket> =
        std::sync::LazyLock::new(|| UdpSocket::bind("127.0.0.1:34254").unwrap());
    static DST_ADDR: SocketAddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 34255);

    eprintln!("send to {DST_ADDR:?}: {result}");
    SOCKET.send_to(&result.into_bytes(), DST_ADDR).unwrap();
}

const NULL: *mut c_void = std::ptr::null_mut();
const PCRE2_ZERO_TERMINATED: PCRE2_SIZE = !0;

fn main() {
    let filename = "input.txt";

    let re = {
        // https://doc.rust-lang.org/std/ffi/struct.CString.html#creating-a-cstring
        let pattern = CString::from(cr"\d{4}([^\d\s]{3,11}).");
        eprintln!("pattern: {pattern:?}");
        let mut errorcode = 0;
        let mut erroroffset: usize = 0;
        // SAFETY: all args are correctly set.
        unsafe {
            pcre2_compile_8(
                pattern.as_ptr().cast(),
                PCRE2_ZERO_TERMINATED,
                0,
                &raw mut errorcode,
                &raw mut erroroffset,
                NULL,
            )
        }
    };
    debug_assert_ne!(re, NULL);

    let f = std::io::BufReader::new(std::fs::File::open(filename).unwrap());
    for (line_no, subject) in f.lines().enumerate() {
        let subject = subject.unwrap();
        let length = subject.len();
        let subject = CString::new(subject).unwrap();
        eprintln!("{line_no} subject: {subject:?}");

        // SAFETY: `re` is valid.
        let match_data = unsafe { pcre2_match_data_create_from_pattern_8(re, NULL) };
        debug_assert!(!match_data.is_null());
        let rc =
            // SAFETY: `re` and `match_data` is valid.
            unsafe { pcre2_match_8(re, subject.as_ptr().cast(), length, 0, 0, match_data, NULL) };
        if rc >= 0 {
            debug_assert_ne!(rc, 0);
            let rc = usize::try_from(rc).unwrap();

            // SAFETY: `match_data` is valid.
            let ovector = unsafe { pcre2_get_ovector_pointer_8(match_data) };
            debug_assert!(!ovector.is_null());

            // SAFETY: `ovector` is valid
            let ovector = unsafe { std::slice::from_raw_parts(ovector, rc * 2) };
            debug_assert!(ovector.len() >= 4);

            eprintln!("Match succeeded at offset {:?}", ovector[0]);

            let substring = {
                let begin = ovector[2];
                let end = ovector[3];
                &subject.as_bytes()[begin..end]
            };
            let result = String::from_utf8(substring.to_owned()).unwrap();
            eprintln!("found: {result}");
            send_result(result);
        }
        // SAFETY: `match_data` is valid
        unsafe {
            pcre2_match_data_free_8(match_data);
        }
    }
    // SAFETY: `re` is valid
    unsafe {
        pcre2_code_free_8(re);
    }
}
