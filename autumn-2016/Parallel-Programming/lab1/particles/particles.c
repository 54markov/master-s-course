#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <xmmintrin.h>
#include <sys/time.h>

#define EPS 1E-6
#define N_STEPS 10

enum {
    //n = 10000
    //n = 1000000
    n = 100000000
};

void init_particles(float *x, float *y, float *z, int n)
{
    for (int i = 0; i < n; i++) {
        x[i] = cos(i + 0.1);
        y[i] = cos(i + 0.2);
        z[i] = cos(i + 0.3);
    }
}

void distance(float *x, float *y, float *z, float *d, int n)
{
    for (int i = 0; i < n; i++) {
        d[i] = sqrtf(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
    }
}

void distance_vec(float *x, float *y, float *z, float *d, int n)
{
    __m128 a, b, c;

    __m128 *xx = (__m128 *)x;
    __m128 *yy = (__m128 *)y;
    __m128 *zz = (__m128 *)z;
    __m128 *dd = (__m128 *)d;

    // Size of vector register is 128 bit 
    int k = n / 4;

    for (int i = 0; i < k; i++) {
        a = _mm_mul_ps(xx[i], xx[i]);
        b = _mm_mul_ps(yy[i], yy[i]);
        c = _mm_mul_ps(zz[i], zz[i]);

        a = _mm_add_ps(a, b);
        c = _mm_add_ps(a, c); 

        dd[i] = _mm_sqrt_ps(c);
    }

    // Loop reminder (n % 4 != 0) ?
    if ((n % 4) != 0) {
        for (int i = (n - (k * 4)); i < n; i++) {
            d[i] = sqrtf(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
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
    float *d, *x, *y, *z;

    x = xmalloc(sizeof(*x) * n);
    y = xmalloc(sizeof(*y) * n);
    z = xmalloc(sizeof(*z) * n);
    d = xmalloc(sizeof(*d) * n);    
    
    init_particles(x, y, z, n);
    
    double t = wtime();
    for (int iter = 0; iter < N_STEPS; iter++) {
        distance(x, y, z, d, n);
    }
    t = (wtime() - t) / N_STEPS;    

    printf("Elapsed time (scalar): %.6f sec.\n", t);
    free(x);
    free(y);    
    free(z);
    free(d);
    return t;
}

double run_vectorized()
{
    float *d, *x, *y, *z;

    x = _mm_malloc(sizeof(*x) * n, 16);
    y = _mm_malloc(sizeof(*y) * n, 16);
    z = _mm_malloc(sizeof(*z) * n, 16);
    d = _mm_malloc(sizeof(*d) * n, 16);
    
    init_particles(x, y, z, n);
    
    double t = wtime();
    for (int iter = 0; iter < N_STEPS; iter++) {
        distance_vec(x, y, z, d, n);
    }
    t = (wtime() - t) / N_STEPS;    
/*
    // Verification
    for (int i = 0; i < n; i++) {
        float x = cos(i + 0.1);
        float y = cos(i + 0.2);
        float z = cos(i + 0.3);
        float dist = sqrtf(x * x + y * y + z * z);
        if (fabs(d[i] - dist) > EPS) {
            fprintf(stderr, "Verification failed: d[%d] = %f != %f\n", i, d[i], dist);
            break;
        }
    }
*/
    printf("Elapsed time (vectorized): %.6f sec.\n", t);
    free(x);
    free(y);    
    free(z);
    free(d);
    return t;
}

int main(int argc, char **argv)
{
    printf("Particles: n = %d)\n", n);
    double tscalar = run_scalar();
    double tvec    = run_vectorized();
    
    printf("Speedup: %.2f\n", tscalar / tvec);
        
    return 0;
}
