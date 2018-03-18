#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>

#include <stdio.h>
#include <cuda.h>
#include <math.h>

/* Every thread gets exactly one value in the unsorted array. */
#define THREADS   1024
#define BLOCKS    32768
#define NUM_VALS  THREADS * BLOCKS

struct Ffunctor
{
    __host__ __device__
    float operator()(float x)
    {
        return sin(x); // change function here
    }
}; 

struct Ifunctor
{
    float h;
    Ifunctor(float _h) : h(_h) { }
    __host__ __device__
    float operator()(float x)
    {
        return h * x;
    }
};

void print_elapsed(clock_t start, clock_t stop)
{
    double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
    printf("Elapsed time: %.3fs\n", elapsed);
}

void thrust_code(const int size)
{
    Ifunctor iFunctor(0.2); // Intialize functor
    Ffunctor fFunctor;      // Functional functor

    // Create vectors on host
    thrust::host_vector<float> hostVector(size);
    thrust::host_vector<float> deviceVector(size);

    // Init vectors on host
    thrust::sequence(hostVector.begin(), hostVector.end());
    thrust::sequence(deviceVector.begin(), deviceVector.end());

    // Run on device
    thrust::transform(deviceVector.begin(),
                      deviceVector.end(),
                      deviceVector.begin(),
                      iFunctor);

    // Run on device
    thrust::transform(deviceVector.begin(),
                      deviceVector.end(),
                      deviceVector.begin(),
                      fFunctor);
    
    // Copy from device to host
    thrust::copy(deviceVector.begin(), deviceVector.end(), hostVector.begin());

    for (int i = 0; i < size - 1; ++i)
    {
        //printf("%04d %f \t %f\n", i, hostVector[i + 1] - hostVector[i], deviceVector[i]);
    }

    for (int i = 0; i < 10; ++i)
    {
        printf("%f %f\n", (float)i, hostVector[i]);
    }
    printf("\n");

    for (int c = 0; c < 10; c++)
    {
        for (int i = 0; i < size - 1; ++i)
        {
            float u = -0.50;
            float t = 1.0;
            float h = 0.99;

            //hostVector[i] = ((u * t) / h) * (hostVector[i + 1] - hostVector[i]);
            hostVector[i] += (hostVector[i + 1] - hostVector[i]) / hostVector[i];
        }

        char name[32] = { 0 };
        sprintf(name, "sin-cuda-%d.txt", c);
        FILE *fp = fopen(name, "w+");
        for (int i = 0; i < size - 1; ++i)
        {
            //printf("%f %f\n", (float)i, hostVector[i]);
            fprintf(fp, "%.6f\t%.6f\n", (float)i * 0.2, hostVector[i]);
        }
        fclose(fp);
        printf("\n");
    }
}

int main(int argc, char const *argv[])
{
    clock_t start, stop;

    start = clock();
    thrust_code(/*NUM_VALS*/ 1 << 8);
    stop = clock();
    
    print_elapsed(start, stop);

    return 0;
}
