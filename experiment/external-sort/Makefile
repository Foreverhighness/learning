CXX = g++
CXXFLAGS = -m64 -std=c++20 -Og -g -Wall -Werror

BUILD_DIR = build

test: $(BUILD_DIR)/main
	./gen.py -r
	$(BUILD_DIR)/main
	bash -c 'diff <(./bin2txt.py $(BUILD_DIR)/1-0.run) <(./bin2txt.py $(BUILD_DIR)/{0..31}.bin | sort -n)'

run: $(BUILD_DIR)/main
	./gen.py -r
	$(BUILD_DIR)/main
	bash -c 'diff <(./bin2txt.py $(BUILD_DIR)/result.out) <(./bin2txt.py $(BUILD_DIR)/*.bin | sort -n)'

$(BUILD_DIR)/main: main.cpp
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $(BUILD_DIR)/main main.cpp

clean:
	rm -rf $(BUILD_DIR)

.PHONY: run clean
