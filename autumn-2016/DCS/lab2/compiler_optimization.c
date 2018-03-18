/*
 * Разработайте программу, выполняющую над двумя векторами вещественных чисел 
 * размером 1024 элемента следующие вычисления:
 * 𝐶𝑖 = √𝐴𝑖 ∗ 𝐵𝑖
 * Используя компилятор GCC получили текст программы на ассемблере (опция -S) 
 * для всех уровней оптимизации кода (опции -O0, -O1, -O2, -O3, -Ofast).
 * Проанализируйте сколько операций выполнится для получения результата 
 * в каждом из способов оптимизации кода.
 */
#include <stdio.h>
#include <math.h>

#define SIZE 1024

void init_vector(double *v)
{
    for (int i = 0; i < SIZE; i++) {
        v[i] = 2.0;
    }
}

void calc_vector(double *a, double *b)
{
    //float c[SIZE] __attribute__((aligned(16)));
    double c[SIZE];

    for (int i = 0; i < SIZE; i++) {
        c[i] = sqrt(a[i] * b[i]);
    }
}

int main(int argc, char const *argv[])
{
    //float A[SIZE] __attribute__((aligned(16)));
    //float B[SIZE] __attribute__((aligned(16)));

    double A[SIZE];
    double B[SIZE];

    init_vector(A);
    init_vector(B);

    calc_vector(A, B);

    return 0;
}