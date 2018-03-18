#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <cuda_runtime.h>

enum { NELEMS = 1024 * 1024 };

double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

__global__ void vec_compute(float *a, const float *b, const float *c, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n)
    {
        a[i] = b[i] + c[i];
        a[i] = a[i] * a[i];
    }
}

int main(void)
{
    double t_cudaMalloc  = 0.0;
    double t_cudaMemcpy  = 0.0;
    double t_cudaCompute = 0.0;

    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;

    size_t size = sizeof(float) * NELEMS;

    //int threadsPerBlock = 256;
    //int threadsPerBlock = 512;

    int threadsPerBlock = 1024;
    int blocksPerGrid   = (NELEMS + threadsPerBlock - 1) / threadsPerBlock;

    /* Allocate vectors on host */
    float *h_A  = (float *)malloc(size);
    float *h_B  = (float *)malloc(size);
    float *h_C  = (float *)malloc(size);

    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Allocation error.\n");
        return -1;
    }

    /* Initialize vectors on host */
    for (int i = 0; i < NELEMS; ++i)
    {
        h_A[i] = 0.0;
        h_B[i] = rand() / (float)RAND_MAX;
        h_C[i] = rand() / (float)RAND_MAX;
    }

    t_cudaMalloc = wtime();
    /* Allocate vectors on device */
    if (cudaMalloc((void **)&d_A, size) != cudaSuccess)
    {
        fprintf(stderr, "Allocation error in cudaMalloc()\n");
        return -1;
    }
    
    if (cudaMalloc((void **)&d_B, size) != cudaSuccess)
    {
        fprintf(stderr, "Allocation error in cudaMalloc()\n");
        return -1;
    }
    
    if (cudaMalloc((void **)&d_C, size) != cudaSuccess)
    {
        fprintf(stderr, "Allocation error in cudaMalloc()\n");
        return -1;
    }
    t_cudaMalloc = wtime() - t_cudaMalloc;

    t_cudaMemcpy = wtime();
    /* Copy the host vectors to device */
    if (cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        fprintf(stderr, "Host to device copying failed\n");
        return -1;
    }

    if (cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        fprintf(stderr, "Host to device copying failed\n");
        return -1;
    }

    if (cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        fprintf(stderr, "Host to device copying failed\n");
        return -1;
    }
    t_cudaMemcpy = wtime() - t_cudaMemcpy;

    t_cudaCompute = wtime();
    /* Launch the kernel */
    vec_compute<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, NELEMS);
    t_cudaCompute = wtime() - t_cudaCompute;
    
    if (cudaGetLastError() != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch kernel!\n");
        return -1;
    }

    /* Copy the device vectors to host */
    if (cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        fprintf(stderr, "Device to host copying failed\n");
        return -1;
    }
/*
    for (int i = 0; i < NELEMS; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
*/

    printf("Vector size                    is %d elements\n", NELEMS);
    printf("Threads per block              is %d elements\n", threadsPerBlock);
    printf("Blocks per grid                is %d elements\n", blocksPerGrid);
    printf("Elapsed time for cudaMalloc()  is %f sec\n", t_cudaMalloc);
    printf("Elapsed time for cudaMalloc()  is %f sec\n", t_cudaMemcpy);
    printf("Elapsed time for cudaCompute() is %f sec\n", t_cudaCompute);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    cudaDeviceReset();

    return 0;
}
