#include <mpi.h>

#include <iostream>
#include <vector>
#include <random>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>
#include <sys/time.h>

#include <omp.h>

using namespace std;

void generatingMatrix(vector<vector<int>> &a, int row, int col)
{
    //cout << "Generating matrix:" << endl;

    for (auto i = 0; i < row; ++i) {
        vector<int> v(col);
        a.push_back(v);
    }
    
    for (auto i = 0; i < row; ++i) {
        for (auto j = 0; j < col; ++j) {
            a[i][j] = j;
        }
    }
/*
    for (auto &i : a) {
        for (auto &j : i) {
            cout << j << " ";
        }
        cout << endl;
    }
*/
}

void multiplyMatrix(vector<vector<int>> &a, vector<vector<int>> &b, int m, int n, int q)
{
    vector<vector<int>> c;

    for (auto i = 0; i < m; ++i) {
        vector<int> v(q);
        c.push_back(v);
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        

        for (auto i = 0; i < m; ++i) {
            for (auto j = 0; j < q; ++j) {
                c[i][j] = 0;
                for (auto k = 0; k < n; ++k) {
                    c[i][j] += a[i][k]*b[k][j];
                }
            }
        }
    } else {
        #pragma omp parallel
        {
            printf("omp_get_num_threads() %d\n", omp_get_num_threads());

            int n_threads        = omp_get_num_threads();
            int thread_id        = omp_get_thread_num();
            int items_per_thread = m / n_threads;

            int low_b = thread_id * items_per_thread;
            int upr_b = (thread_id == n_threads - 1) ? (m - 1) : (low_b + items_per_thread - 1);

            for (int i = low_b; i <= upr_b; i++) {
                for (auto j = 0; j < q; ++j) {
                    c[i][j] = 0;
                    for (auto k = 0; k < n; ++k) {
                        c[i][j] += a[i][k]*b[k][j];
                    }
                }
            }
        }
    }
/*
    sleep(rank);

    printf("Rank %d:\n", rank);

    for (auto i = 0; i < m; ++i) {
        for (auto j = 0; j < q; ++j) {
            cout << c[i][j] << " ";
        }
        cout << endl;
    }
*/
}

double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

void run_matrix()
{
    vector<vector<int>> matrix1;
    vector<vector<int>> matrix2;

    int size = 1024;

    generatingMatrix(matrix1, 1024, 1024);
    generatingMatrix(matrix2, 1024, 1024);

    multiplyMatrix(matrix1, matrix2, 1024, 1024, 1024);
}

int main(int argc, char** argv)
{
    int world_size, world_rank;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    double t = wtime();
    run_matrix();
    printf("Rank %d elapsed time %f sec\n", world_rank, wtime() - t);

    MPI_Finalize();
    return 0;
}
