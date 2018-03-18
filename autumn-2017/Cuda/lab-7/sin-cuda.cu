#include <cuda.h>
#include <math.h>
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

__global__ void kernel(float *a, float *b, const int n)
{
    float u = -0.50;
    float t = 1.0;
    float h = 0.99;

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if ((i + 1) < n)
    {
        a[i] += ((u * t) / h) * (a[i + 1] - a[i]); 
    }
}

float init_function(float x) 
{
    return sin(x * 0.02);
}

int main(int argc, char **argv)
{
    const int num_of_blocks     = 10;
    const int threads_per_block = 32;
    const int vector_size       = num_of_blocks * threads_per_block;
    const int bytes             = vector_size * sizeof(float);

    int devId = 0;
    cudaDeviceProp properties;
    cudaEvent_t    start_event;
    cudaEvent_t    stop_event;

    float *host_vector1   = NULL;
    float *host_vector2   = NULL;
    float *device_vector1 = NULL;
    float *device_vector2 = NULL;

    float ms = 0.0; // Elapsed time in milliseconds

    checkCuda(cudaGetDeviceProperties(&properties, devId));
    printf("Device : %s\n", properties.name);
    checkCuda(cudaSetDevice(devId));

    // Allocate memory for the host vector
    host_vector1 = (float *)malloc(bytes);
    if (!host_vector1)
    {
        fprintf(stderr, "Can't allocate memory for vector\n");
        exit(EXIT_FAILURE);
    }
    
    host_vector2 = (float *)malloc(bytes);
    if (!host_vector1)
    {
        fprintf(stderr, "Can't allocate memory for vector\n");
        exit(EXIT_FAILURE);
    }

    // Initialize host vector
    for (int i = 0; i < vector_size; i++)
    {
        host_vector1[i] = init_function((float)i);
        host_vector2[i] = init_function((float)i);
    }

    // Allocate memory for the device vector
    checkCuda(checkCuda(cudaMalloc((void**)&device_vector1, bytes)));
    checkCuda(checkCuda(cudaMalloc((void**)&device_vector2, bytes)));

    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    // Run it
    checkCuda(cudaEventRecord(start_event, 0));
    for (int i = 0; i < 50; i++)
    {
        // Copy to device (gpu)
        checkCuda(cudaMemcpy(device_vector1, host_vector1, bytes, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(device_vector2, host_vector2, bytes, cudaMemcpyHostToDevice));
        kernel<<<dim3(num_of_blocks), dim3(threads_per_block)>>>(device_vector1, device_vector2, vector_size);
        checkCuda(cudaDeviceSynchronize());
        // Copy from device (gpu)
        checkCuda(cudaMemcpy(host_vector1, device_vector1, bytes, cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(host_vector2, device_vector2, bytes, cudaMemcpyDeviceToHost));

        char name[32] = { 0 };
        sprintf(name, "sin-cuda-%d.txt", i);
        FILE *fp = fopen(name, "w+");
        for (int j = 0; j < vector_size; j++)
        {
            //printf("%.6f\t%.6f\t%.6f\n", (float)j * 0.02, host_vector1[j], host_vector2[j]);
            fprintf(fp, "%.6f\t%.6f\n", (float)j * 0.02, host_vector1[j]);
            //host_vector1[j] = host_vector2[j];
        }
        //printf("\n");
        fclose(fp);
    }
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&ms, start_event, stop_event);

    printf("Running cuda-sin implementation\n");
    printf("Elapsed time: %f\n", ms);

    checkCuda(cudaFree(device_vector1));
    checkCuda(cudaFree(device_vector2));
    free(host_vector1);
    free(host_vector2);

    return 0;
}
