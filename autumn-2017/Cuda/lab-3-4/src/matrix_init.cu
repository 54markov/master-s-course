#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <malloc.h>

__global__ void matrix_init_by_row(float* a, int N)
{
    int thread_id=threadIdx.x+blockIdx.x*blockDim.x;
    
    for(int i = thread_id; i < N; i += blockDim.x * gridDim.x)
    {
        a[i] = i;
    }
}

__global__ void matrix_init_by_col(float* a, int N)
{
    int thread_id = threadIdx.y + blockIdx.y * blockDim.y;
    
    for(int i = thread_id; i < N; i += blockDim.y * gridDim.y)
    {
        a[i] = i;
    }
}

int main(int argc, char* argv[])
{
    float *dev_matrix = NULL;

    if (argc < 4)
    {
        fprintf(stderr, "USAGE: %s <vector length> <blocks>  <threads>\n", argv[0]);
        return -1;
    }

    int N = atoi(argv[1]);
    int num_of_blocks = atoi(argv[2]);
    int threads_per_block = atoi(argv[3]);

    cudaMalloc((void**)&dev_matrix, N * sizeof(float));

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);
    matrix_init_by_row<<<num_of_blocks, threads_per_block>>>(dev_matrix, N);
    cudaThreadSynchronize();
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime,start,stop);
    printf("matrix_init_by_row took %g\n", elapsedTime);

    cudaEventRecord(start,0);  
    matrix_init_by_col<<<num_of_blocks, threads_per_block>>>(dev_matrix, N);
    cudaThreadSynchronize();
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime,start,stop); 
    printf("matrix_init_by_col took %g\n", elapsedTime);
  
    cudaFree(dev_matrix);
  
    return 0;
}
