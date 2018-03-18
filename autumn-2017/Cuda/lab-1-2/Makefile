all: test

1: 1.cu
	nvcc -g -o 1 1.cu

2: 2.cu
	nvcc -g -o 2 2.cu

3: 3.cu
	nvcc -g -o 3 3.cu

vec_compute: vec_compute.cu
	nvcc -g -o vec_compute vec_compute.cu

vec_add: vec_add.cu
	nvcc -g -o vec_add vec_add.cu

test: vec_add.cu
	nvcc -g -o test_vadd1 vec_add.cu -D TEST_SIZE_1 -D TEST_CONF_1
	nvcc -g -o test_vadd2 vec_add.cu -D TEST_SIZE_2 -D TEST_CONF_1
	nvcc -g -o test_vadd3 vec_add.cu -D TEST_SIZE_3 -D TEST_CONF_1
	nvcc -g -o test_vadd4 vec_add.cu -D TEST_SIZE_1 -D TEST_CONF_2
	nvcc -g -o test_vadd5 vec_add.cu -D TEST_SIZE_2 -D TEST_CONF_2
	nvcc -g -o test_vadd6 vec_add.cu -D TEST_SIZE_3 -D TEST_CONF_2

clean:
	rm -f 1 2 3 vec_compute test_vadd*
