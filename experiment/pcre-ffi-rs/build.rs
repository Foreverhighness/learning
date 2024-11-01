use std::io::Cursor;
use std::path::{Path, PathBuf};

use bzip2::read::BzDecoder;
use cmake::Config;
use http_req::request;
use sha2::{Digest, Sha256};
use tar::Archive;

const DOWNLOAD_BASE_URL: &str = "https://github.com/PCRE2Project/pcre2/releases/download";
const PCRE2: &str = "pcre2";
const VERSION: &str = "10.42";
const SHA256: &str = "8d36cd8cb6ea2a4c2bb358ff6411b0c788633a2a45dabbf1aeb4b701d1b5e840";

const LIB_NAME: &str = "pcre2-8";
const BUILD_TARGET: &str = "pcre2-8-static";

fn main() {
    let path = get_pcre2();
    let dst = Config::new(path).build_target(BUILD_TARGET).build();
    println!(
        "cargo:rustc-link-search=native={}",
        dst.join("build").display()
    );
    println!("cargo:rustc-link-lib=static={LIB_NAME}");
    println!("cargo:rerun-if-changed=build.rs");
}

fn get_pcre2() -> PathBuf {
    let out_dir = std::env::var_os("OUT_DIR").unwrap();
    let out_dir = Path::new(&out_dir);
    let source_dir = out_dir.join("source");

    let compressed_file = download_compressed_file();

    let mut archive = Archive::new(BzDecoder::new(compressed_file));
    archive.unpack(&source_dir).unwrap();

    let mut source_dir = source_dir;
    let basename = format!("{PCRE2}-{VERSION}");
    source_dir.push(basename);
    source_dir
}

fn try_download(url: &str) -> Result<Cursor<Vec<u8>>, String> {
    let body = {
        let mut body = Vec::new();
        let mut url = url.to_owned();
        for _retry in 0..3 {
            // Send GET request
            let response = request::get(&url, &mut body).map_err(|e| e.to_string())?;
            if response.status_code().is_redirect() {
                url = response.headers().get("Location").cloned().unwrap();
                continue;
            }
            if response.status_code().is_success() {
                break;
            }
            return Err(format!("Download error: HTTP {}", response.status_code()));
        }
        body
    };

    // Check the SHA-256 hash of the downloaded file is as expected
    let hash = Sha256::digest(&body);
    if format!("{hash:x}") != SHA256 {
        return Err(format!("Downloaded {PCRE2} file failed hash check."));
    }
    Ok(Cursor::new(body))
}

fn download_compressed_file() -> Cursor<Vec<u8>> {
    let basename = format!("{PCRE2}-{VERSION}");
    let filename = format!("{basename}.tar.bz2");
    let url = format!("{DOWNLOAD_BASE_URL}/{basename}/{filename}");

    try_download(&url).unwrap_or_else(|e| panic!("\n\nDownload error: {e}\n\n"))
}
