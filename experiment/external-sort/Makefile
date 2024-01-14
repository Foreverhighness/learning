CXX = clang++
CXXFLAGS = -m64 -std=c++20 -Og -g -Wall -Werror
LDLIBS = -lstdc++fs

BUILD_DIR = build

run: $(BUILD_DIR)/main
	./gen.py -r
	$(BUILD_DIR)/main
	bash -c 'diff <(./bin2txt.py $(BUILD_DIR)/result.out) <(./bin2txt.py $(BUILD_DIR)/*.bin | sort -n)'

$(BUILD_DIR)/main: main.cpp
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: run clean
