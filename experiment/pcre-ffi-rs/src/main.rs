#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
use std::ffi::{c_void, CString};
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
    static CELL: std::sync::OnceLock<UdpSocket> = std::sync::OnceLock::new();

    let socket = CELL.get_or_init(|| UdpSocket::bind("127.0.0.1:34254").unwrap());
    let port = 34255;
    let dst = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), port);
    println!("send to {dst:?}: {result}");
    socket.send_to(&result.into_bytes(), dst).unwrap();
}

const NULL: *mut c_void = std::ptr::null::<c_void>().cast_mut();
const PCRE2_ZERO_TERMINATED: PCRE2_SIZE = !0;
use std::io::BufRead;

fn main() {
    let filename = "input.txt";

    unsafe {
        let re = {
            let pattern = CString::new(r"\d{4}([^\d\s]{3,11}).").unwrap();
            println!("pattern: {pattern:?}");
            let mut errorcode = 0;
            let mut erroroffset: usize = 0;
            pcre2_compile_8(
                pattern.as_ptr().cast(),
                PCRE2_ZERO_TERMINATED,
                0,
                &mut errorcode as _,
                &mut erroroffset as _,
                NULL,
            )
        };
        debug_assert_ne!(re, NULL);

        let f = std::io::BufReader::new(std::fs::File::open(filename).unwrap());
        for (line, subject) in f.lines().enumerate() {
            let subject = subject.unwrap();
            let length = subject.len();
            let subject = CString::new(subject.into_bytes()).unwrap();
            println!("{line} subject: {subject:?}");

            let match_data = pcre2_match_data_create_from_pattern_8(re, NULL);
            let rc = pcre2_match_8(re, subject.as_ptr().cast(), length, 0, 0, match_data, NULL);
            if rc >= 0 {
                debug_assert_ne!(rc, 0);
                let rc: usize = rc.try_into().unwrap();

                let ovector = pcre2_get_ovector_pointer_8(match_data);
                let ovector = std::slice::from_raw_parts(ovector, rc * 2);
                debug_assert!(ovector.len() >= 4);
                println!("Match succeeded at offset {:?}", ovector.get_unchecked(0));

                let substring_start = subject.as_ptr().add(*ovector.get_unchecked(2)).cast();
                let substring_length = ovector.get_unchecked(3) - ovector.get_unchecked(2);
                let slice = std::slice::from_raw_parts(substring_start, substring_length);
                let result = String::from_utf8_unchecked(slice.to_owned());
                println!("found: {result}");
                send_result(result);
            }
            pcre2_match_data_free_8(match_data);
        }
        pcre2_code_free_8(re);
    };
}
