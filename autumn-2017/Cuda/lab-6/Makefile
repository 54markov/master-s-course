CC := nvcc
CFLAGS := -ccbin clang-3.8 -Wno-deprecated-gpu-targets -arch=sm_21

all: copy-gpu-bandwith vector-add-stream vector-mul-stream

vector-add-stream: vector-add-stream.cu
	mkdir -p build
	$(CC) $(CFLAGS) -g -o build/vector-add-stream vector-add-stream.cu

vector-mul-stream: vector-mul-stream.cu
	mkdir -p build
	$(CC) $(CFLAGS) -g -o build/vector-mul-stream vector-mul-stream.cu

copy-gpu-bandwith: copy-gpu-bandwith.cu
	mkdir -p build
	$(CC) $(CFLAGS) -g -o build/copy-gpu-bandwith copy-gpu-bandwith.cu

clean:
	rm -Rf build
