#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

#include <xmmintrin.h> /* SSE, нужен также mmintrin.h */
#include <emmintrin.h> /* SSE2, нужен также xmmintrin.h */
#include <pmmintrin.h> /* SSE3, нужен также emmintrin.h */
#include <smmintrin.h> /* SSE4.1 */
#include <nmmintrin.h> /* SSE4.2 */

#define N_STEPS 10
#define EPS 1E-6

enum { n = 1000007 };

void compute_sqrt(float *in, float *out, int n)
{
    for (int i = 0; i < n; i++) {
        if (in[i] > 0) {
            out[i] = sqrtf(in[i]);
        } else {
            out[i] = 0.0;
        }
    }
}

void compute_sqrt_sse(float *in, float *out, int n)
{
	__m128 *in_vec  = (__m128 *)in;
    __m128 *out_vec = (__m128 *)out;
    __m128 zero     = _mm_setzero_ps();

    // Size of vector register is 128 bit 
    int k = n / 4;
    
    for (int i = 0; i < k; i++) {
        __m128 v          = _mm_load_ps((float *)&in_vec[i]);
        __m128 sqrt_vec   = _mm_sqrt_ps(v);
        __m128 mask       = _mm_cmpgt_ps(v, zero);
        __m128 gtzero_vec = _mm_and_ps(mask, sqrt_vec);
        __m128 lezero_vec = _mm_andnot_ps(mask, zero);
        out_vec[i]        = _mm_or_ps(gtzero_vec, lezero_vec);
    }

    // Loop reminder (n % 4 != 0) ?
    if ((n % 4) != 0) {
        for (int i = k * 4; i < n; i++) {
            out[i] = in[i] > 0 ? sqrtf(in[i]) : 0.0;
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

double run_scalar()
{
    float *in = xmalloc(sizeof(*in) * n);
    float *out = xmalloc(sizeof(*out) * n);
    srand(0);
    for (int i = 0; i < n; i++) {
        in[i] = rand() > RAND_MAX / 2 ? 0 : rand() / (float)RAND_MAX * 1000.0;
    }
    
    //warm-up
    compute_sqrt(in, out, n);

    double t = wtime();
    for (int i = 0; i < N_STEPS; i++) {
        compute_sqrt(in, out, n);
    }
    t = wtime() - t;
    t /= N_STEPS;

    printf("Elapsed time (scalar)    : %.6f sec.\n", t);
    free(in);
    free(out);
    return t;
}

double run_vectorized()
{
    float *in  = _mm_malloc(sizeof(*in) * n, 16);
    float *out = _mm_malloc(sizeof(*out) * n, 16);
    srand(0);
    for (int i = 0; i < n; i++) {
        in[i] = rand() > RAND_MAX / 2 ? 0 : rand() / (float)RAND_MAX * 1000.0;
    }

    //warm-up
    compute_sqrt_sse(in, out, n);
    for (int i = 0; i < n; i++) {
        float r = in[i] > 0 ? sqrtf(in[i]) : 0.0;
        if (fabs(out[i] - r) > EPS) {
            fprintf(stderr, "Verification: FAILED at out[%d] = %f != %f\n", i, out[i], r);
            break;
        }
    }

    double t = wtime();
    for (int i = 0; i < N_STEPS; i++) {
        compute_sqrt_sse(in, out, n);
    }
    t = wtime() - t;
    t /= N_STEPS;

    printf("Elapsed time (vectorized): %.6f sec.\n", t);
    free(in);
    free(out);
    return t;
}

int main(int argc, char **argv)
{
    printf("Tabulate sqrt sse        : n = %d\n", n);
    double tscalar = run_scalar();
    double tvec = run_vectorized();
    
    printf("Speedup                  : %.2f\n", tscalar / tvec);
        
    return 0;
}
