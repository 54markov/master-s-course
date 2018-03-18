#include <mpi.h>

#include <iostream>
#include <vector>
#include <random>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>
#include <sys/time.h>

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

    sleep(rank);

    printf("Rank %d:\n", rank);  
    
    for (auto i = 0; i < m; ++i) {
        for (auto j = 0; j < q; ++j) {
            c[i][j] = 0;
            
            for (auto k = 0; k < n; ++k) {
                c[i][j] = c[i][j] + (a[i][k]*b[k][j]);
            }
            cout << c[i][j] << " ";
        }
        cout << endl;
    }
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

    generatingMatrix(matrix1, 10, 10);
    generatingMatrix(matrix2, 10, 10);

    multiplyMatrix(matrix1, matrix2, 10, 10, 10);
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
