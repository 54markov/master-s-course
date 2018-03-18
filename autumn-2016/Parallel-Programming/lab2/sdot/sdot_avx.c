#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include <immintrin.h> /* AVX */

#define N_STEPS 10

enum { n = 1000003 };

float sdot(float *x, float *y, int n)
{
    float s = 0;

    for (int i = 0; i < n; i++) {
        s += x[i] * y[i];
    }

    return s;
}

float sdot_avx(float *x, float *y, int n)
{
/*
    float s[8] __attribute__ ((aligned (32)));
    float result = 0;
*/
    __m256 *xx  = (__m256 *)x;
    __m256 *yy  = (__m256 *)y;
    __m256 ss   = _mm256_setzero_ps();
    __m256 temp = _mm256_setzero_ps();
   
    // Size of vector register is 256 bit 
    int k = n / 8;
    
    for (int i = 0; i < k; i++) {
        temp = _mm256_mul_ps(xx[i], yy[i]);
        ss = _mm256_add_ps(temp, ss);
    }
/*
    _mm256_store_ps(s, ss);
    result = s[0] + s[1] + s[2] + s[3] + s[4] + s[5] + s[6] + s[7];
*/
    // Compute s[0] + s[1] + s[2] + s[3] + s[4] + s[5] + s[6] + s[7]
    ss = _mm256_hadd_ps(ss, ss);
    ss = _mm256_hadd_ps(ss, ss);
    // Permute high and low bits
    temp = _mm256_permute2f128_ps(ss, ss, 1);
    ss = _mm256_add_ps(ss, temp);
    float t[8] __attribute__ ((aligned (32)));
    _mm256_store_ps(t, ss); 
    float result = t[0];
/*
    temp = _mm256_permute2f128_ps(ss , ss , 1);
    ss = _mm256_add_ps(ss, temp);
    ss = _mm256_hadd_ps(ss, ss);
    ss = _mm256_hadd_ps(ss, ss);

    float t[8] __attribute__ ((aligned (32)));
    _mm256_store_ps(t, ss); 
    float result = t[0];
*/
    // Loop reminder (n % 8 != 0) ?
    if ((n % 8) != 0) {
        for (int i = (k * 8); i < n; i++) {
            result += x[i] * y[i];
        }
    }
    return result;
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

double run_scalar()
{
    float *x = xmalloc(sizeof(*x) * n);
    float *y = xmalloc(sizeof(*y) * n);
    for (int i = 0; i < n; i++) {
        x[i] = 2.0;
        y[i] = 3.0;
    }
    
    // warm-up
    float res = sdot(x, y, n);
    float valid_result = 2.0 * 3.0 * (float)n;
    printf("Result (scalar)          : %.6f err = %f\n", res, fabsf(valid_result - res));
    
    double t = wtime();
    for (int i = 0; i < N_STEPS; i++) {
        sdot(x, y, n);
    }
    t = wtime() - t;
    t /= N_STEPS;
  
    printf("Elapsed time (scalar)    : %.6f sec.\n", t);
    free(x);
    free(y);
    return t;
}

double run_vectorized()
{
    float *x = _mm_malloc(sizeof(*x) * n, 32);
    float *y = _mm_malloc(sizeof(*y) * n, 32);

    for (int i = 0; i < n; i++) {
        x[i] = 2.0;
        y[i] = 3.0;
    }        
    
    // warm-up
    float res = sdot_avx(x, y, n);
    float valid_result = 2.0 * 3.0 * (float)n;
    printf("Result (vectorized)      : %.6f err = %f\n", res, fabsf(valid_result - res));

    double t = wtime();
    for (int i = 0; i < N_STEPS; i++) {
        sdot_avx(x, y, n);
    }
    t = wtime() - t;    
    t /= N_STEPS;

    printf("Elapsed time (vectorized): %.6f sec.\n", t);
    free(x);
    free(y);
    return t;
}

int main(int argc, char **argv)
{
    printf("SDOT avx                 : n = %d\n", n);
    float tscalar = run_scalar();
    float tvec = run_vectorized();
    
    printf("Speedup avx              : %.2f\n", tscalar / tvec);
        
    return 0;
}