#include <cuda.h>
#include <stdio.h>
#include <assert.h>

const int TILE_DIM   = 32;
const int BLOCK_ROWS = 8;
const int NUM_REPS   = 100;

/*
 * Check errors and print GB/s
 */
void post_process(const float *ref, const float *res, int n, float ms)
{
    bool passed = true;

    for (int i = 0; i < n; i++)
    {
        if (res[i] != ref[i])
        {
            printf("%d %f %f\n", i, res[i], ref[i]);
            printf("%25s\n", "*** FAILED ***");
            passed = false;
            break;
        }
    }

    if (passed)
    {
        printf("%20.2f\n", 2 * n * sizeof(float) * 1e-6 * NUM_REPS / ms );
    }
}

/*
 * Simple copy kernel:
 * Used as reference case representing best effective bandwidth.
 */
__global__ void copy_simple(float *odata, const float *idata)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
    {
        odata[(y + j) * width + x] = idata[(y + j) * width + x];
    }
}

/*
 * Copy kernel using shared memory:
 * Also used as reference case, demonstrating effect of using shared memory.
 */
__global__ void copy_shared_mem(float *odata, const float *idata)
{
    __shared__ float tile[TILE_DIM * TILE_DIM];  // Shared memory

    int x     = blockIdx.x * TILE_DIM + threadIdx.x;
    int y     = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        tile[(threadIdx.y + j) * TILE_DIM + threadIdx.x] = idata[(y + j) * width + x];
    }

    __syncthreads(); // Threads block barrier synchronization

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        odata[(y + j) * width + x] = tile[(threadIdx.y + j) * TILE_DIM + threadIdx.x];          
    }
}

/*
 * Naive transpose:
 * Simplest transpose; doesn't use shared memory.
 * Global memory reads are coalesced but writes are not.
 */
__global__ void transpose_naive(float *odata, const float *idata)
{
    int x     = blockIdx.x * TILE_DIM + threadIdx.x;
    int y     = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
    {
        odata[x * width + (y + j)] = idata[(y + j) * width + x];
    }
}

/*
 * Coalesced transpose:
 * Uses shared memory to achieve coalesing in both reads and writes
 * Tile width == #banks causes shared memory bank conflicts.
 */
__global__ void transpose_coalesced(float *odata, const float *idata)
{
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x     = blockIdx.x * TILE_DIM + threadIdx.x;
    int y     = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
    }

    __syncthreads(); // Threads block barrier synchronization

    x = blockIdx.y * TILE_DIM + threadIdx.x; // Transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        odata[(y + j)  *width + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}
   
/*
 * No bank-conflict transpose
 * Same as transpose coalesced, except the first tile dimension is padded 
 * to avoid shared memory bank conflicts.
 */
__global__ void transpose_no_bank_conflicts(float *odata, const float *idata)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // Shared memory

    int x     = blockIdx.x * TILE_DIM + threadIdx.x;
    int y     = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
    }

    __syncthreads(); // Threads block barrier synchronization

    x = blockIdx.y * TILE_DIM + threadIdx.x; // Transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        odata[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

int main(int argc, char **argv)
{
    const int nx = /*1024;*/4096;
    const int ny = /*1024;*/4096;
    const int mem_size = nx * ny * sizeof(float);

    dim3 dimGrid(nx / TILE_DIM, ny / TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

    int devId = 0;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, devId);
    printf("\nDevice : %s\n", prop.name);
    printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n", 
            nx, ny, TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);

    printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n",
            dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);

    cudaSetDevice(devId);

    float *h_idata = (float *)malloc(mem_size);
    float *h_cdata = (float *)malloc(mem_size);
    float *h_tdata = (float *)malloc(mem_size);
    float *gold    = (float *)malloc(mem_size);

    float *d_idata, *d_cdata, *d_tdata;
    cudaMalloc(&d_idata, mem_size);
    cudaMalloc(&d_cdata, mem_size);
    cudaMalloc(&d_tdata, mem_size);

    // Check parameters and calculate execution configuration
    if (nx % TILE_DIM || ny % TILE_DIM)
    {
        fprintf(stderr, "nx and ny must be a multiple of TILE_DIM\n");
        cudaFree(d_tdata);
        cudaFree(d_cdata);
        cudaFree(d_idata);
        free(h_idata);
        free(h_tdata);
        free(h_cdata);
        free(gold);

        exit(EXIT_FAILURE);
    }

    if (TILE_DIM % BLOCK_ROWS)
    {
        fprintf(stderr, "TILE_DIM must be a multiple of BLOCK_ROWS\n");
        cudaFree(d_tdata);
        cudaFree(d_cdata);
        cudaFree(d_idata);
        free(h_idata);
        free(h_tdata);
        free(h_cdata);
        free(gold);

        exit(EXIT_FAILURE);
    }

    // Host init
    int i, j;
    for (j = 0; j < ny; j++)
    {
        for (i = 0; i < nx; i++)
        {
            h_idata[j * nx + i] = j * nx + i;
        }
    }

    // Correct result for error checking
    for (j = 0; j < ny; j++)
    {
        for (i = 0; i < nx; i++)
        {
            gold[j * nx + i] = h_idata[i * nx + j];
        }
    }

    // Device
    cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice);
    // Events for timing
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float ms;

    // Time kernels
    printf("%25s%25s\n", "Routine", "Bandwidth (GB/s)");

    // Copy
    printf("%25s", "copy");
    cudaMemset(d_cdata, 0, mem_size);
    
    // Warm up
    copy_simple<<<dimGrid, dimBlock>>>(d_cdata, d_idata);

    cudaEventRecord(startEvent, 0);
    for (int i = 0; i < NUM_REPS; i++)
    {
        copy_simple<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
    }
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost);
    post_process(h_idata, h_cdata, nx * ny, ms);


    // Copy Shared Memory
    printf("%25s", "shared memory copy");
    cudaMemset(d_cdata, 0, mem_size);

    // Warm up
    copy_shared_mem<<<dimGrid, dimBlock>>>(d_cdata, d_idata);

    cudaEventRecord(startEvent, 0);
    for (i = 0; i < NUM_REPS; i++)
    {
        copy_shared_mem<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
    }
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost);
    post_process(h_idata, h_cdata, nx * ny, ms);

    // Transpose Naive 
    printf("%25s", "naive transpose");
    cudaMemset(d_tdata, 0, mem_size);

    // Warmup
    transpose_naive<<<dimGrid, dimBlock>>>(d_tdata, d_idata);

    cudaEventRecord(startEvent, 0);
    for (i = 0; i < NUM_REPS; i++)
    {
        transpose_naive<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    }
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost);
    post_process(gold, h_tdata, nx * ny, ms);

    // TransposeCoalesced 
    printf("%25s", "coalesced transpose");
    cudaMemset(d_tdata, 0, mem_size);

    // warmup
    transpose_coalesced<<<dimGrid, dimBlock>>>(d_tdata, d_idata);

    cudaEventRecord(startEvent, 0);
    for (int i = 0; i < NUM_REPS; i++)
    {
        transpose_coalesced<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    }
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost);
    post_process(gold, h_tdata, nx * ny, ms);

    // TransposeNoBankConflicts
    printf("%25s", "conflict-free transpose");
    cudaMemset(d_tdata, 0, mem_size);

    // Warmup
    transpose_no_bank_conflicts<<<dimGrid, dimBlock>>>(d_tdata, d_idata);

    cudaEventRecord(startEvent, 0);
    for (i = 0; i < NUM_REPS; i++)
    {
        transpose_no_bank_conflicts<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    }
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost);
    post_process(gold, h_tdata, nx * ny, ms);

    // Cleanup
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_tdata);
    cudaFree(d_cdata);
    cudaFree(d_idata);
    free(h_idata);
    free(h_tdata);
    free(h_cdata);
    free(gold);
    return 0;
}
