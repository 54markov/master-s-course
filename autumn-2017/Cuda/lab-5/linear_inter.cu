/*
 * https://codeplea.com/simple-interpolation
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define M_PI            3.14159265358979323846
#define COEF            48
#define VERTCOUNT       (COEF * COEF * 2)
#define RADIUS          10.0f
#define FGSIZE          20
#define FGSHIFT         (FGSIZE / 2)
#define IMIN(A,B)       (A < B ? A : B)
#define THREADSPERBLOCK 256
#define BLOCKSPERGRID   IMIN(32, (VERTCOUNT + THREADSPERBLOCK - 1) / THREADSPERBLOCK)

typedef float(*ptr_f)(float, float, float);

struct Vertex
{
    float x, y, z;
};

float func(float x, float y, float z)
{
    return ((0.5 * sqrtf(15.0 / M_PI)) * (0.5 * sqrtf(15.0 / M_PI)) *
             z * z * y * y * sqrtf(1.0f - z * z / RADIUS / RADIUS) /
             RADIUS / RADIUS / RADIUS / RADIUS);
}

float host_check(Vertex *v, ptr_f f)
{
    float sum = 0.0f;
    
    for (int i = 0; i < VERTCOUNT; ++i)
    {
        sum += f(v[i].x, v[i].y, v[i].z);
    }

    return sum;
}

void host()
{
    Vertex *temp_vert = (Vertex *)malloc(sizeof(Vertex) * VERTCOUNT);
    
    int i = 0;
    for (int iphi = 0; iphi < 2 * COEF; ++iphi)
    {
        for (int ipsi = 0; ipsi < COEF; ++ipsi, ++i)
        {
            float phi      = iphi * M_PI / COEF;
            float psi      = ipsi * M_PI / COEF;
            temp_vert[i].x = RADIUS * sinf(psi) * cosf(phi);
            temp_vert[i].y = RADIUS * sinf(psi) * sinf(phi);
            temp_vert[i].z = RADIUS * cosf(psi);
        }
    }

    printf("sumcheck = %f\n", host_check(temp_vert, &func) * M_PI * M_PI / COEF / COEF);

    free(temp_vert);
}

__global__ void kernel(float *sum_dev)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < VERTCOUNT)
    {

        int iphi = tid / (2 * COEF);
        int ipsi = tid - (iphi * (2 * COEF));

        float phi = iphi * M_PI / COEF;
        float psi = ipsi * M_PI / COEF;
       
        Vertex vert = {
            .x = RADIUS * sinf(psi) * cosf(phi),
            .y = RADIUS * sinf(psi) * sinf(phi),
            .z = RADIUS * cosf(psi)
        };

        vert.y -= 0.5f;
        vert.z -= 0.5f;

        float f0 = ((0.5 * sqrtf(15.0 / M_PI)) * (0.5 * sqrtf(15.0 / M_PI)) *
                         vert.z * vert.z * vert.y * vert.y *
                         sqrtf(1.0f - vert.z * vert.z / RADIUS / RADIUS) /
                         RADIUS / RADIUS / RADIUS / RADIUS);
        
        Vertex vert1 = {
            .x = RADIUS * sinf(psi) * cosf(phi),
            .y = RADIUS * sinf(psi) * sinf(phi),
            .z = RADIUS * cosf(psi)
        };

        vert1.y += 0.5f;
        vert1.z += 0.5f;

        float f1 = ((0.5 * sqrtf(15.0 / M_PI)) * (0.5 * sqrtf(15.0 / M_PI)) *
                         vert1.z * vert1.z * vert1.y * vert1.y *
                         sqrtf(1.0f - vert1.z * vert1.z / RADIUS / RADIUS) /
                         RADIUS / RADIUS / RADIUS / RADIUS);

        float xL = tid;
        float yL = f0;

        float xR = tid + 1;
        float yR = f1;

        float dydx = (yR - yL) / (xR - xL); // Gradient

        sum_dev[tid] = yL + dydx * ((float)tid - xL); // Linear interpolation

        //sum_dev[tid] = f1;
    }
    __syncthreads();
}


int main(int argc, char const *argv[])
{
    cudaEvent_t start;
    cudaEvent_t stop;

    float  t_cudaCompute = 0.0;

    float *sum     = (float*)malloc(sizeof(float) * VERTCOUNT);
    float *sum_dev = NULL;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc((void**)&sum_dev, sizeof(float) * VERTCOUNT);

    host();

    // Launch the kernel
    cudaEventRecord(start, 0);
    kernel<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(sum_dev);
    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_cudaCompute, start, stop);

    cudaMemcpy(sum, sum_dev, sizeof(float) * VERTCOUNT, cudaMemcpyDeviceToHost);

    float s = 0.0f;

    for (int i = 0; i < VERTCOUNT; ++i)
    {
        if (sum[i] != sum[i])
        {
            sum[i] = 0.0;
        }
        s += sum[i];
        //printf("%f ", sum[i]);
    }

    printf("sum = %f\n", s * M_PI* M_PI / COEF / COEF);

    printf("Elapsed time is %f sec\n", t_cudaCompute);

    cudaFree(sum_dev);
    free(sum);

    return 0;
}
