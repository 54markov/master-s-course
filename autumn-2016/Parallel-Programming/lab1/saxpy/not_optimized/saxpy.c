#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <xmmintrin.h>
#include <sys/time.h>

#define EPS 1E-6
#define N_STEPS 10

enum 
{
    n = 100000000
    //n = 10000000 
    //n = 1000000 
    //n = 100000
    //n = 10000
    //n = 1000
};

void saxpy(float *x, float *y, float a, int n)
{
    for (int i = 0; i < n; i++)
        y[i] = a * x[i] + y[i];
}

void saxpy_sse(float * restrict x, float * restrict y, float a, int n)
{
    __m128 *xx = (__m128 *)x;
    __m128 *yy = (__m128 *)y;
   
   // Size of vector register is 128 bit 
    int k = n / 4;
    
    __m128 aa = _mm_set1_ps(a);
    
    for (int i = 0; i < k; i++) {
        __m128 z = _mm_mul_ps(aa, xx[i]);          
        yy[i] = _mm_add_ps(z, yy[i]);
    }

    // Loop reminder (n % 4 != 0) ?
    if ((n % 4) != 0) {
        for (int i = (n - (k * 4)); i < n; i++) {
            y[i] = a * x[i] + y[i];
        }
    }
}

void daxpy(double *x, double *y, double a, int n)
{
    for (int i = 0; i < n; i++)
        y[i] = a * x[i] + y[i];
}

void daxpy_sse(double * restrict x, double * restrict y, double a, int n)
{
    __m128d *xx = (__m128d *)x;
    __m128d *yy = (__m128d *)y;
   
    // Size of vector register is 128 bit
    int k = n / 2;
    
    __m128d aa = _mm_set1_pd(a);
    
    for (int i = 0; i < k; i++) {
        __m128d z = _mm_mul_pd(aa, xx[i]);          
        yy[i] = _mm_add_pd(z, yy[i]);
    }

    // Loop reminder (n % 2 != 0) ?
    if ((n % 2) != 0) {
        for (int i = (n - (k * 2)); i < n; i++) {
            y[i] = a * x[i] + y[i];
        }
    }
}

void *xmalloc(size_t size)
{
    void *p = malloc(size);
    if (!p) {
        fprintf(stderr, "malloc failed\n");
        exit(EXIT_FAILURE);
    }
    return p;
}

double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

double run_scalar_daxpy()
{
    double *x, *y, a = 2.0;

    x = xmalloc(sizeof(*x) * n);
    y = xmalloc(sizeof(*y) * n);

    // Initialize only once
    for (int i = 0; i < n; i++) {
        x[i] = i * 2 + 1.0;
        y[i] = i;
    }
    
    // Warm-up
    daxpy(x, y, a, n);
    
    double t = wtime();
    for (int i = 0; i < N_STEPS; i++) {
        daxpy(x, y, a, n);
    }
    t = (wtime() - t) / N_STEPS;    
/*
    // Verification
    for (int i = 0; i < n; i++) {
        float xx = i * 2 + 1.0;
        float yy = a * xx + i;
        if (fabs(y[i] - yy) > EPS) {
            fprintf(stderr, "run_scalar: verification failed (y[%d] = %f != %f)\n", i, y[i], yy);
            break;
        }
    }
*/  
    printf("Elapsed time (scalar)    : %.6f sec.\n", t);
    free(x);
    free(y);    
    return t;
}

double run_vectorized_daxpy()
{
    double *x, *y, a = 2.0;

    x = _mm_malloc(sizeof(*x) * n, 16); // allign memory by 16
    y = _mm_malloc(sizeof(*y) * n, 16); // allign memory by 16
    for (int i = 0; i < n; i++) {
        x[i] = i * 2 + 1.0;
        y[i] = i;
    }

    // Warm-up
    daxpy_sse(x, y, a, n);

    double t = wtime();
    for (int i = 0; i < N_STEPS; i++) {
        daxpy_sse(x, y, a, n);
    }
    t = (wtime() - t) / N_STEPS;
/*
    // Verification
    for (int i = 0; i < n; i++) {
        double xx = i * 2 + 1.0;
        double yy = a * xx + i;
        if (fabs(y[i] - yy) > EPS) {
            fprintf(stderr, "run_vectorized: verification failed (y[%d] = %f != %f)\n", i, y[i], yy);
            break;
        }
    }
*/
    printf("Elapsed time (vectorized): %.6f sec.\n", t);
    free(x);
    free(y);
    return t;
}

double run_scalar_saxpy()
{
    float *x, *y, a = 2.0;

    x = xmalloc(sizeof(*x) * n);
    y = xmalloc(sizeof(*y) * n);

    // Initialize only once
    for (int i = 0; i < n; i++) {
        x[i] = i * 2 + 1.0;
        y[i] = i;
    }
    
    // Warm-up
    saxpy(x, y, a, n);
    
    double t = wtime();
    for (int i = 0; i < N_STEPS; i++) {
        saxpy(x, y, a, n);
    }
    t = (wtime() - t) / N_STEPS;    
/*
    // Verification
    for (int i = 0; i < n; i++) {
        float xx = i * 2 + 1.0;
        float yy = a * xx + i;
        if (fabs(y[i] - yy) > EPS) {
            fprintf(stderr, "run_scalar: verification failed (y[%d] = %f != %f)\n", i, y[i], yy);
            break;
        }
    }
*/  
    printf("Elapsed time (scalar)    : %.6f sec.\n", t);
    free(x);
    free(y);    
    return t;
}

double run_vectorized_saxpy()
{
    float *x, *y, a = 2.0;

    x = _mm_malloc(sizeof(*x) * n, 16);
    y = _mm_malloc(sizeof(*y) * n, 16);
    for (int i = 0; i < n; i++) {
        x[i] = i * 2 + 1.0;
        y[i] = i;
    }
    
    // Warm-up
    saxpy_sse(x, y, a, n);

    double t = wtime();
    for (int i = 0; i < N_STEPS; i++) {
        saxpy_sse(x, y, a, n);
    }
    t = (wtime() - t ) / N_STEPS;
/*    
    // Verification
    for (int i = 0; i < n; i++) {
        float xx = i * 2 + 1.0;
        float yy = a * xx + i;
        if (fabs(y[i] - yy) > EPS) {
            fprintf(stderr, "run_vectorized: verification failed (y[%d] = %f != %f)\n", i, y[i], yy);
            break;
        }
    }
*/
    printf("Elapsed time (vectorized): %.6f sec.\n", t);
    free(x);
    free(y);
    return t;
}

int main(int argc, char **argv)
{
    double tscalar, tvec;

    printf("SAXPY (y[i] = a * x[i] + y[i]; n = %d)\n", n);
    tscalar = run_scalar_saxpy();
    tvec    = run_vectorized_saxpy();
    printf("Speedup: %.2f\n", tscalar / tvec);
    
    printf("DAXPY (y[i] = a * x[i] + y[i]; n = %d)\n", n);
    tscalar = run_scalar_daxpy();
    tvec    = run_vectorized_daxpy();
    printf("Speedup: %.2f\n", tscalar / tvec);
        
    return 0;
}