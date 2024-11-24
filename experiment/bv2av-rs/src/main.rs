#![feature(str_split_remainder)]

mod flags;

fn parse(bv: &str) -> (&str, &str) {
    let mut bv_parts = bv.rsplit('/');
    let mut bvid = "";
    for bv in bv_parts.by_ref() {
        if bv.starts_with("BV") {
            bvid = bv;
            break;
        }
    }
    let uri = bv_parts.remainder().unwrap_or_default();
    (uri, bvid)
}

fn main() {
    let flags::Cli { bv } = flags::Cli::from_env_or_exit();
    let bv = bv.into_string().unwrap();

    let (uri, bvid) = parse(&bv);

    let slash = if uri.is_empty() { "" } else { "/" };

    let avid = abv::bv2av(bvid).unwrap();
    println!("{uri}{slash}av{avid}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_success() {
        let tests = [
            (
                "https://www.bilibili.com/video/BV1D84118741",
                ("https://www.bilibili.com/video", "BV1D84118741"),
            ),
            (
                "https://www.bilibili.com/video/BV1D84118741/",
                ("https://www.bilibili.com/video", "BV1D84118741"),
            ),
            (
                "https://www.bilibili.com/video/BV1D84118741/vd=114514",
                ("https://www.bilibili.com/video", "BV1D84118741"),
            ),
            (
                "https://www.bilibili.com/video/BV1D84118741/?vd=114514",
                ("https://www.bilibili.com/video", "BV1D84118741"),
            ),
            ("BV1D84118741/", ("", "BV1D84118741")),
        ];

        for (input, (expect_uri, expect_bvid)) in tests {
            let (uri, bvid) = parse(input);
            assert_eq!(uri, expect_uri);
            assert_eq!(bvid, expect_bvid);
        }
    }
}
