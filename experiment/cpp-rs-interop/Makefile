CXX = g++
CXXFLAGS = -m64 -Og -g -Wall -Werror -std=c++20

SRCS_DIR := src
BUILD_DIR := build
CPP_HEADER := $(BUILD_DIR)/user-rs-lib.h
RUST_LIB := $(SRCS_DIR)/lib.rs

SRCS := $(shell find $(SRCS_DIR) -name "*.cpp")
RUNS := $(notdir $(basename $(SRCS)))
EXES := $(RUNS:%=$(BUILD_DIR)/%)
LIB := $(BUILD_DIR)/libuser_rs.a

# $(info $(SRCS) $(RUNS) $(EXES))

run: $(RUNS)

all: $(EXES)

$(RUNS): %: $(BUILD_DIR)/%
	-$<

$(BUILD_DIR)/%: $(SRCS_DIR)/%.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/user-using-rs-lib: $(SRCS_DIR)/user-using-rs-lib.cpp $(CPP_HEADER) $(LIB)
	$(CXX) $(CXXFLAGS) -I$(BUILD_DIR) $^ -o $@

$(CPP_HEADER): $(RUST_LIB)
	-@cargo install cbindgen > /dev/null 2>&1
	cbindgen -v $^ --lang=c++ -o $@

header: $(CPP_HEADER)

$(LIB): $(RUST_LIB)
	cargo build -Z unstable-options --artifact-dir $(BUILD_DIR)

clean:
	cargo clean
	rm -rf $(BUILD_DIR)

.PHONY: run all header clean
