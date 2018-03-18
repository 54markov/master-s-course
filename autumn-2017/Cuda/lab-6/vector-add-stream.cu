#include <cuda.h>
#include <stdio.h>

inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
    return result;
}

__global__ void kernel(float *a, float *b, int offset)
{
    int i = offset + threadIdx.x + blockIdx.x*blockDim.x;
    a[i] = a[i] + b[i];
}

float validation(float *a, int n) 
{
    float result = 0.0;
    for (int i = 0; i < n; i++)
    {
        result += a[i];
    }
    return result;
}

int main(int argc, char **argv)
{
    const int blockSize   = 512;
    const int nStreams    = 8;
    const int n           = 2 * 1024 * blockSize * nStreams;
    const int streamSize  = n / nStreams;
    const int streamBytes = streamSize * sizeof(float);
    const int bytes       = n * sizeof(float);

    int devId = 0;

    cudaDeviceProp prop;

    checkCuda( cudaGetDeviceProperties(&prop, devId));
    printf("Device : %s\n", prop.name);
    checkCuda( cudaSetDevice(devId) );

    // Allocate pinned host memory and device memory
    float *array = NULL;
    float *a     = NULL;
    float *b     = NULL;
    float *d_a   = NULL;
    float *d_b   = NULL;

    checkCuda(cudaMallocHost((void**)&a, bytes)); // Host pinned
    checkCuda(cudaMallocHost((void**)&b, bytes)); // Host pinned
    checkCuda(cudaMalloc((void**)&d_a, bytes));   // Device
    checkCuda(cudaMalloc((void**)&d_b, bytes));   // Device

    array = (float *)malloc(bytes);

    for (int i = 0; i < n; i++)
    {
        array[i] = (float)i;
    }

    float ms = 0.0; // Elapsed time in milliseconds

    // Create events and streams
    cudaEvent_t startEvent, stopEvent, dummyEvent;
    cudaStream_t stream[nStreams];

    checkCuda(cudaEventCreate(&startEvent));
    checkCuda(cudaEventCreate(&stopEvent));
    checkCuda(cudaEventCreate(&dummyEvent));

    for (int i = 0; i < nStreams; ++i)
    {
        checkCuda(cudaStreamCreate(&stream[i]));
    }

    // baseline case - sequential transfer and execute
    memcpy(a, array, bytes);
    memcpy(b, array, bytes);

    checkCuda(cudaEventRecord(startEvent,0));
    checkCuda(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

    kernel<<<n / blockSize, blockSize>>>(d_a, d_b, 0);

    checkCuda(cudaMemcpy(a, d_a, bytes, cudaMemcpyDeviceToHost));
    checkCuda(cudaEventRecord(stopEvent, 0) );
    checkCuda(cudaEventSynchronize(stopEvent) );
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));

    printf("Time for sequential transfer and execute (ms): %f\n", ms);
    printf("  validation: %e\n", validation(a, n));

    // Asynchronous version: loop over { copy, kernel, copy }
    memcpy(a, array, bytes);
    memcpy(b, array, bytes);

    checkCuda(cudaEventRecord(startEvent,0));
    for (int i = 0; i < nStreams; ++i)
    {
        int offset = i * streamSize;

        checkCuda(cudaMemcpyAsync(&d_a[offset], &a[offset],
            streamBytes, cudaMemcpyHostToDevice, stream[i]));

        checkCuda(cudaMemcpyAsync(&d_b[offset], &b[offset],
            streamBytes, cudaMemcpyHostToDevice, stream[i]));
    }

    for (int i = 0; i < nStreams; ++i)
    {
        int offset = i * streamSize;
        kernel<<<streamSize / blockSize, blockSize, 0, stream[i]>>>(d_a, d_b, offset);
    }

    for (int i = 0; i < nStreams; ++i)
    {
        int offset = i * streamSize;
        checkCuda(cudaMemcpyAsync(&a[offset], &d_a[offset],
            streamBytes, cudaMemcpyDeviceToHost, stream[i]));
    }
    checkCuda( cudaEventRecord(stopEvent, 0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    printf("Time for asynchronous transfer and execute (ms): %f\n", ms);
    printf("  validation: %e\n", validation(a, n));

    for (int i = 0; i < n; i++)
    {
        array[i] = array[i] + array[i];
    }

    printf("  validation: %e\n", validation(a, n));

    // Cleanup
    checkCuda(cudaEventDestroy(startEvent));
    checkCuda(cudaEventDestroy(stopEvent));
    checkCuda(cudaEventDestroy(dummyEvent));

    for (int i = 0; i < nStreams; ++i)
    {
        checkCuda( cudaStreamDestroy(stream[i]));
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFreeHost(a);
    cudaFreeHost(b);

    free(array);

    return 0;
}
