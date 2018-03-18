#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <cuda_runtime.h>

#ifdef TEST_SIZE_1
enum { NELEMS = 1<<16 };
#endif /* SIZE_TEST_1 */

#ifdef TEST_SIZE_2
enum { NELEMS = 1<<20 };
#endif /* SIZE_TEST_2 */

#ifdef TEST_SIZE_3
enum { NELEMS = 1<<24 };
#endif /* SIZE_TEST_3 */

//enum { NELEMS = 1024 * 1024 };

double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

__global__ void vadd(const float *a, const float *b, float *c, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

int main(void)
{
    cudaEvent_t start;
    cudaEvent_t stop;

    double t_cudaMalloc  = 0.0;
    double t_cudaMemcpy  = 0.0;
    float  t_cudaCompute = 0.0;

    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;

    size_t size = sizeof(float) * NELEMS;

    int threadsPerBlock = 1024;

#ifdef TEST_CONF_1
    threadsPerBlock = 512;
#endif /* TEST_CONF_1 */

#ifdef TEST_CONF_2
    threadsPerBlock = 1024;
#endif /* TEST_CONF_2 */

    int blocksPerGrid = (NELEMS + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate vectors on host
    float *h_A  = (float *)malloc(size);
    float *h_B  = (float *)malloc(size);
    float *h_C  = (float *)malloc(size);

    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Host Allocation error\n");
        return -1;
    }

    // Initialize vectors on host
    for (int i = 0; i < NELEMS; ++i)
    {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
        h_C[i] = 0.0;
    }

    // Allocate vectors on device
    t_cudaMalloc = wtime();
    if (cudaMalloc((void **)&d_A, size) != cudaSuccess)
    {
        fprintf(stderr, "Allocation error in cudaMalloc()\n");
        exit(EXIT_FAILURE);
    }
    
    if (cudaMalloc((void **)&d_B, size) != cudaSuccess)
    {
        fprintf(stderr, "Allocation error in cudaMalloc()\n");
        exit(EXIT_FAILURE);
    }
    
    if (cudaMalloc((void **)&d_C, size) != cudaSuccess)
    {
        fprintf(stderr, "Allocation error in cudaMalloc()\n");
        exit(EXIT_FAILURE);
    }
    t_cudaMalloc = wtime() - t_cudaMalloc;

    // Copy the host vectors to device
    t_cudaMemcpy = wtime();
    if (cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        fprintf(stderr, "Host to device copying failed\n");
        exit(EXIT_FAILURE);
    }

    if (cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        fprintf(stderr, "Host to device copying failed\n");
        exit(EXIT_FAILURE);
    }

    if (cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        fprintf(stderr, "Host to device copying failed\n");
        exit(EXIT_FAILURE);
    }
    t_cudaMemcpy = wtime() - t_cudaMemcpy;

    // Launch the kernel
    cudaEventRecord(start, 0);
    vadd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, NELEMS);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_cudaCompute, start, stop);

    if (cudaGetLastError() != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch kernel!\n");
        exit(EXIT_FAILURE);
    }

    // Copy the device vectors to host
    if (cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        fprintf(stderr, "Device to host copying failed\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < NELEMS; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Vector size                    is %d elements\n", NELEMS);
    printf("Threads per block              is %d elements\n", threadsPerBlock);
    printf("Blocks per grid                is %d elements\n", blocksPerGrid);
    //printf("Elapsed time for cudaMalloc()  is %f sec\n", t_cudaMalloc);
    //printf("Elapsed time for cudaMemcpy()  is %f sec\n", t_cudaMemcpy);
    printf("Elapsed time for cudaCompute() is %f sec\n", t_cudaCompute);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceReset();

    return 0;
}
