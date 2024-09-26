CC = gcc
CFLAGS = -m64 -std=c17 -O3

BUILD_DIR := target/release

run: all
	@$(BUILD_DIR)/thread-local-rs
	@$(BUILD_DIR)/thread-local-c

all: $(BUILD_DIR)/thread-local-rs $(BUILD_DIR)/thread-local-c

$(BUILD_DIR)/thread-local-rs: src/main.rs build.rs src/thread_local.c
	cargo build --release

$(BUILD_DIR)/thread-local-c: src/main.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $^ -o $@

clean:
	cargo clean

.PHONY: run all clean
