#include <cuda.h>
#include <stdio.h>

/*
 * Convenience function for checking CUDA runtime API results
 * can be wrapped around any runtime API call. No-op in release builds.
 */
inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
    return result;
}

void profileCopies(float *h_a, float *h_b, float *d, 
                   const unsigned int n, char *desc)
{
    float time = 0.0;
    const unsigned int bytes = n * sizeof(float);

    // Events for timing
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;

    checkCuda(cudaEventCreate(&startEvent));
    checkCuda(cudaEventCreate(&stopEvent));

    printf("\n%s transfers\n", desc);

    checkCuda(cudaEventRecord(startEvent, 0));
    checkCuda(cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&time, startEvent, stopEvent));

    printf("  Host to Device bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

    checkCuda(cudaEventRecord(startEvent, 0));
    checkCuda(cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost));
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&time, startEvent, stopEvent));

    printf("  Device to Host bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

    for (int i = 0; i < n; ++i)
    {
        if (h_a[i] != h_b[i])
        {
            printf("*** %s transfers failed ***", desc);
            break;
        }
    }

    // Clean up events
    checkCuda(cudaEventDestroy(startEvent));
    checkCuda(cudaEventDestroy(stopEvent));
}

int main(int argc, char const *argv[])
{
    const unsigned int nElements = 16 * 1024 * 1024;
    const unsigned int bytes     = nElements * sizeof(float);

    // Host arrays
    float *h_aPageable = NULL;
    float *h_bPageable = NULL;
    float *h_aPinned   = NULL;
    float *h_bPinned   = NULL;

    // Device array
    float *d_a = NULL;

    // Allocate and initialize
    h_aPageable = (float*)malloc(bytes); // Host pageable
    h_bPageable = (float*)malloc(bytes); // Host pageable

    checkCuda(cudaMallocHost((void**)&h_aPinned, bytes)); // Host pinned
    checkCuda(cudaMallocHost((void**)&h_bPinned, bytes)); // Host pinned

    checkCuda(cudaMalloc((void**)&d_a, bytes)); // Device

    for (int i = 0; i < nElements; ++i)
    {
        h_aPageable[i] = i;
    }

    memcpy(h_aPinned,   h_aPageable, bytes);
    memset(h_bPageable, 0,           bytes);
    memset(h_bPinned,   0,           bytes);

    // Output device info and transfer size
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, 0));

    printf("\nDevice: %s\n", prop.name);
    printf("Transfer size (MB): %d\n", bytes / (1024 * 1024));

    // Perform copies and report bandwidth
    profileCopies(h_aPageable, h_bPageable, d_a, nElements, (char *)"Pageable");
    profileCopies(h_aPinned, h_bPinned, d_a, nElements, (char *)"Pinned");

    printf("\n");

    // Cleanup
    cudaFree(d_a);
    cudaFreeHost(h_aPinned);
    cudaFreeHost(h_bPinned);
    free(h_aPageable);
    free(h_bPageable);

    return 0;
}
