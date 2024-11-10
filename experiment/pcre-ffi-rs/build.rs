use std::io::Cursor;
use std::path::{Path, PathBuf};

use bzip2::read::BzDecoder;
use http_req::request;
use sha2::{Digest, Sha256};
use tar::Archive;

const DOWNLOAD_BASE_URL: &str = "https://github.com/PCRE2Project/pcre2/releases/download";
const PCRE2: &str = "pcre2";
const VERSION: &str = "10.42";
const PCRE2_PATH: &str = "pcre2-10.42";
const SHA256: &str = "8d36cd8cb6ea2a4c2bb358ff6411b0c788633a2a45dabbf1aeb4b701d1b5e840";

const LIB_BASENAME: &str = "pcre2-8";
const LIB_NAME: &str = "libpcre2-8";
const BUILD_TARGET: &str = "pcre2-8-static";

fn main() {
    if let Err(_) = pkg_config::Config::new()
        .statik(true)
        .atleast_version(VERSION)
        .probe(LIB_NAME)
    {
        let build_dir = build_pcre2();
        println!("cargo:rustc-link-search=native={}", build_dir.display());
        println!("cargo:rustc-link-lib=static={LIB_BASENAME}");
    }

    println!("cargo:rerun-if-changed=build.rs");
}

fn build_pcre2() -> PathBuf {
    let compressed_file = download_compressed_file();
    let mut archive = Archive::new(BzDecoder::new(Cursor::new(compressed_file)));

    let out_dir = std::env::var_os("OUT_DIR").unwrap();
    let out_dir = Path::new(&out_dir);
    let third_party = out_dir.join("third_party");
    archive.unpack(&third_party).unwrap();

    let source_dir = {
        let mut third_party = third_party;
        third_party.push(PCRE2_PATH);
        third_party
    };

    let dst_dir = (cmake::Config::new(source_dir))
        .build_target(BUILD_TARGET)
        .build();

    let build_dir = {
        let mut dst_dir = dst_dir;
        dst_dir.push("build");
        dst_dir
    };

    build_dir
}

fn try_download(url: &str) -> Result<Vec<u8>, String> {
    let body = {
        let mut body = Vec::new();
        let mut url = url.to_owned();
        for _retry in 0..3 {
            // Send GET request
            let response = request::get(&url, &mut body).map_err(|e| e.to_string())?;
            let status_code = response.status_code();
            if status_code.is_redirect() {
                url = response.headers().get("Location").cloned().unwrap();
                continue;
            }
            if status_code.is_success() {
                break;
            }
            return Err(format!("Download error: HTTP {status_code}"));
        }
        body
    };

    // Check the SHA-256 hash of the downloaded file is as expected
    let hash = Sha256::digest(&body);
    if format!("{hash:x}") != SHA256 {
        return Err(format!("Downloaded {PCRE2} file failed hash check."));
    }

    Ok(body)
}

fn download_compressed_file() -> Vec<u8> {
    let filename = format!("{PCRE2_PATH}.tar.bz2");
    let url = format!("{DOWNLOAD_BASE_URL}/{PCRE2_PATH}/{filename}");

    try_download(&url).unwrap_or_else(|e| panic!("\n\nDownload error: {e}\n\n"))
}
