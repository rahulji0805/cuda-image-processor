NVCC        = nvcc
NVCC_FLAGS  = -O2 -arch=sm_50 -std=c++14 -Isrc
TARGET      = image_processor
SRC         = src/image_processor.cu

.PHONY: all clean run generate

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)

generate:
	python3 scripts/generate_test_images.py

run: $(TARGET)
	mkdir -p results data/output
	./$(TARGET) $(ARGS)

clean:
	rm -f $(TARGET)
	rm -f data/output/*.pgm
	rm -f results/*.csv results/*.png

build: all
