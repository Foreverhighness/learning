run: ../dll/main ../target/debug/libld_preload.so
	LD_PRELOAD=../target/debug/libld_preload.so ../dll/main

check: ../target/debug/libld_preload.so
	nm --defined-only --dynamic ../target/debug/libld_preload.so

../target/debug/libld_preload.so: src/lib.rs
	cargo build

../dll/main: ../dll/main.c
	$(MAKE) -C ../dll

clean:
	$(MAKE) -C ../dll clean

.PHONY: run check clean