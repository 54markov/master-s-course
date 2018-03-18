CC := nvcc
CFLAGS := -ccbin clang-3.8 -Wno-deprecated-gpu-targets -arch=sm_21

all: task1 step_inter linear_inter

task1: task1.cu
	mkdir -p build
	$(CC) $(CFLAGS) -g -o build/task1 task1.cu -lm

step_inter: step_inter.cu
	mkdir -p build
	$(CC) $(CFLAGS) -g -o build/step_inter step_inter.cu -lm

linear_inter: linear_inter.cu
	mkdir -p build
	$(CC) $(CFLAGS) -g -o build/linear_inter linear_inter.cu -lm

clean:
	rm -Rf build
