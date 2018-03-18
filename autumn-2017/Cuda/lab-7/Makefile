CC       := nvcc
CXXFLAGS := -ccbin clang++-3.8 -Wno-deprecated-gpu-targets -arch=sm_21
CFLAGS   := -ccbin clang-3.8 -Wno-deprecated-gpu-targets -arch=sm_21

all: thrust

test: test.cu
	mkdir -p build
	$(CC) $(CXXFLAGS) -g -o build/test test.cu

sin-cuda: sin-cuda.cu
	mkdir -p build
	$(CC) $(CFLAGS) -g -o build/sin-cuda sin-cuda.cu -lm

thrust: thrust.cu
	mkdir -p build
	$(CC) $(CXXFLAGS) -g -o build/thrust thrust.cu -lm

clean:
	rm -Rf build
