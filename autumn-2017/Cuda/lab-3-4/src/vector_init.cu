#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void vector_init(int *v)
{
    v[threadIdx.x + blockDim.x * blockIdx.x] = (int)(threadIdx.x + blockDim.x * blockIdx.x);
}

void test(int num_threads, int num_blocks)
{
    int *v_device = NULL;
    int *v_host   = NULL;

    cudaEvent_t start;
    cudaEvent_t stop;

    float t_cudaCompute = 0.0;

    int size = num_threads * num_blocks;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    v_host = (int *)malloc(sizeof(int) * size);
    if (!v_host)
    {
        fprintf(stderr, "Host Allocation error\n");
        exit(EXIT_FAILURE);
    }

    if (cudaMalloc((void **)&v_device, (sizeof(int) * size)) != cudaSuccess)
    {
        fprintf(stderr, "Allocation error in cudaMalloc()\n");
        exit(EXIT_FAILURE);
    }
    
    // Launch the kernel
    cudaEventRecord(start, 0);
    vector_init<<<num_blocks, num_threads>>>(v_device);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_cudaCompute, start, stop);

    printf("Elapsed time for vector_init() is %f sec (warm up)\n", t_cudaCompute);

    // Launch the kernel
    cudaEventRecord(start, 0);
    vector_init<<<num_threads, num_blocks>>>(v_device);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_cudaCompute, start, stop);

    printf("Elapsed time for vector_init() is %f sec\n", t_cudaCompute);

    cudaFree(v_device);
    free(v_host);

    return;
}

int main(int argc, char const *argv[])
{
    if (argc < 3)
    {
        fprintf(stderr, "Running auto configuration...\n");

        test(512, 512);
        test(512, 256);
        test(512, 192);
        test(512, 128);
        test(512, 64);
        test(512, 48);
        test(512, 32);
        test(512, 512);
        test(256, 512);
        test(192, 512);
        test(128, 512);
        test(64, 512);
        test(48, 512);
        test(32, 512);
    }
    else
    {
        int num_threads = atoi(argv[1]);
        int num_blocks  = atoi(argv[2]);

        fprintf(stderr, "Running manual configuration...\n");

        test(num_threads, num_blocks);
    }

    return 0;
}
