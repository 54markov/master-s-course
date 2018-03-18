CC := nvcc
CFLAGS := -ccbin clang-3.8 -Wno-deprecated-gpu-targets -arch=sm_21

all: vector_init matrix_transpose matrix_init

vector_init: vector_init.cu
	$(CC) $(CFLAGS) -g -o vector_init vector_init.cu

matrix_transpose: matrix_transpose.cu
	$(CC) $(CFLAGS) -g -o matrix_transpose matrix_transpose.cu

matrix_init: matrix_init.cu
	$(CC) $(CFLAGS)  -g -o matrix_init matrix_init.cu

clean:
	rm -f vector_init matrix_transpose matrix_init
