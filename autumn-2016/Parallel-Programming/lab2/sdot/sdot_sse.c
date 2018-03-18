#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include <xmmintrin.h> /* SSE, нужен также mmintrin.h */
#include <emmintrin.h> /* SSE2, нужен также xmmintrin.h */
#include <pmmintrin.h> /* SSE3, нужен также emmintrin.h */
#include <smmintrin.h> /* SSE4.1 */
#include <nmmintrin.h> /* SSE4.2 */

#define N_STEPS 10

enum { n = 1000003 };

float sdot(float *x, float *y, int n)
{
    float s = 0;
    for (int i = 0; i < n; i++)
        s += x[i] * y[i];
    return s;
}

float sdot_sse(float *x, float *y, int n)
{
    float s[4] __attribute__ ((aligned (16)));
    //float result = 0;

    __m128 *xx  = (__m128 *)x;
    __m128 *yy  = (__m128 *)y;
    __m128 ss   = _mm_setzero_ps();
    __m128 temp = _mm_setzero_ps();
   
    // Size of vector register is 128 bit 
    int k = n / 4;
    
    for (int i = 0; i < k; i++) {
        temp = _mm_mul_ps(xx[i], yy[i]);
        ss = _mm_add_ps(temp, ss);
    }

    //_mm_store_ps(s, ss);
    //result = s[0] + s[1] + s[2] + s[3];

    // s = sumv[0] + sumv[1] + sumv[2] + sumv[3]
    ss = _mm_hadd_ps(ss, ss);
    ss = _mm_hadd_ps(ss, ss);
    float result __attribute__ ((aligned (16))) = 0;
    _mm_store_ss(&result, ss);

    // Loop reminder (n % 4 != 0) ?
    if ((n % 4) != 0) {
        for (int i = (k * 4); i < n; i++) {
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
    float valid_result = 2.0 * 3.0 * n;
    printf("Result (scalar)          : %.6f err = %f\n", res, fabsf(valid_result - res));
    
    double t = wtime();
    for (int i = 0; i < N_STEPS; i++) {
        float res = sdot(x, y, n);
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
    float *x = _mm_malloc(sizeof(*x) * n, 16);
    float *y = _mm_malloc(sizeof(*y) * n, 16);

    for (int i = 0; i < n; i++) {
        x[i] = 2.0;
        y[i] = 3.0;
    }

    // warm-up
    float res = sdot_sse(x, y, n);
    float valid_result = 2.0 * 3.0 * n;
    printf("Result (vectorized)      : %.6f err = %f\n", res, fabsf(valid_result - res));
    
    double t = wtime();
    for (int i = 0; i < N_STEPS; i++) {
        sdot_sse(x, y, n);
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
    printf("SDOT sse                 : n = %d\n", n);
    float tscalar = run_scalar();
    float tvec = run_vectorized();
    
    printf("Speedup sse              : %.2f\n", tscalar / tvec);
        
    return 0;
}
