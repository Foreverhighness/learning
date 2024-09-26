CXX = g++
CXXFLAGS = -m64 -Og -g -Wall -Werror -std=c++20

SRCS_DIR := src
BUILD_DIR := build
CPP_HEADER := $(SRCS_DIR)/user-rs-lib.h
RUST_LIB := $(SRCS_DIR)/lib.rs

SRCS := $(shell find $(SRCS_DIR) -name "*.cpp")
RUNS := $(notdir $(basename $(SRCS)))
EXES := $(RUNS:%=$(BUILD_DIR)/%)
LIB := "target/debug/libuser_rs.a"

# $(info $(SRCS) $(RUNS) $(EXES))

run: $(RUNS)

all: $(EXES)

$(RUNS): %: $(BUILD_DIR)/%
	-$<

$(BUILD_DIR)/%: $(SRCS_DIR)/%.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/user-using-rs-lib: $(SRCS_DIR)/user-using-rs-lib.cpp $(CPP_HEADER) $(LIB)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(CPP_HEADER): $(RUST_LIB)
	-@cargo install cbindgen > /dev/null 2>&1
	cbindgen -v $^ --lang=c++ -o $@

header: $(CPP_HEADER)

$(LIB): $(RUST_LIB)
	cargo build

clean:
	cargo clean
	rm -r $(BUILD_DIR)
	rm $(CPP_HEADER)

.PHONY: run all header clean
