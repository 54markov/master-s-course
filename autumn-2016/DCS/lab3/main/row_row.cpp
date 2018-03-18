#include <iostream>
#include <vector>
#include <random>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fstream>
#include <string>

#include <ctime>
#include <ratio>
#include <chrono>

#include <omp.h>

using namespace std;

vector<vector<int>> c1;
vector<vector<int>> c2;

void generatingMatrix(vector<vector<int>> &a, int row, int col)
{
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, 100);

    //cout << "Generating matrix:" << endl;

    for (auto i = 0; i < row; ++i) {
        vector<int> v(col);
        a.push_back(v);
    }
    
    for (auto i = 0; i < row; ++i) {
        for (auto j = 0; j < col; ++j) {
            a[i][j] = distribution(generator);
        }
    }
    
    for (auto i = 0; i < row; ++i) {
        vector<int> v(col);
        c1.push_back(v);
        c2.push_back(v);
    }
/*
    for (auto &i : a) {
        for (auto &j : i) {
            cout << j << "\t";
        }
        cout << endl;
    }
*/
}

void multiplyMatrix(vector<vector<int>> &a, vector<vector<int>> &b, int m, int n, int q)
{
    //cout << "The new matrix is:" << endl;
    
    for (auto i = 0; i < m; ++i) {
        for (auto k = 0; k < q; ++k) {
            for (auto j = 0; j < n; ++j) {
                c1[i][j] = c1[i][j] + (a[i][k]*b[k][j]);
            }
        }
        
    }
/*
    for (auto i = 0; i < m; ++i) {
        for (auto k = 0; k < q; ++k) {
            cout << c[i][k] << " ";
        }
        cout << endl;
    }
*/
}

void multiplyMatrixOMP(vector<vector<int>> &a, vector<vector<int>> &b, int m, int n, int q)
{
    //cout << "C = A * B" << endl;
    //cout << "Result matrix is:" << endl;

    int i, j, k;

    #pragma omp parallel shared(a,b,c2) private(i,j,k)
    {
        std::cout << __FILE__<< " omp_get_num_threads()           : "<< omp_get_num_threads() << std::endl;
        #pragma omp for schedule(static)
        for (i = 0; i < m; ++i) {
            for (k = 0; k < q; ++k) {
                for (j = 0; j < n; ++j) {
                    c2[i][j] = c2[i][j] + (a[i][k]*b[k][j]);
                }
            }
        }
    }
/*
    for (auto i = 0; i < m; ++i) {
        for (auto k = 0; k < q; ++k) {
            cout << c[i][k] << "\t";
        }
        cout << endl;
    }
*/
}

int verfication(int m, int q)
{
    for (auto i = 0; i < m; ++i) {
        for (auto k = 0; k < q; ++k) {
            if (c1[i][k] != c2[i][k]) {
                std::cout << __FILE__<< " c1[" << i << "][" << k << "] != c2[" << i << "][" << k << "] " << c1[i][k] << " != " << c2[i][k] << std::endl;
                return 0;
            }
        }
    }
    return 1;
}
 
int main(int argc, char const *argv[])
{
    std::string line;

    std::vector<vector<int>> matrix1;
    std::vector<vector<int>> matrix2;
        
    int param = 256;
        
    generatingMatrix(matrix1, param, param);
    generatingMatrix(matrix2, param, param);

    chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
        
    multiplyMatrix(matrix1, matrix2, param, param, param);
        
    chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
    chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    std::cout << __FILE__<< " multiplyMatrix() elapsed time   : " << time_span.count() << " seconds." << std::endl;

    chrono::high_resolution_clock::time_point t3 = chrono::high_resolution_clock::now();
    multiplyMatrixOMP(matrix1, matrix2, param, param, param);
    
    chrono::high_resolution_clock::time_point t4 = chrono::high_resolution_clock::now();
    chrono::duration<double> time_span1 = chrono::duration_cast<chrono::duration<double>>(t4 - t3);
    std::cout << __FILE__<< " multiplyMatrixOMP() elapsed time: "<< time_span1.count() << " seconds." << std::endl;

    std::cout << __FILE__<< " Speedup matrix size " << param << "         : "<< time_span.count() / time_span1.count() << std::endl;
    std::cout << __FILE__<< " omp_get_num_threads()           : "<< omp_get_num_threads() << std::endl;

    if (verfication(param, param)) {
        std::cout << __FILE__<< " verfication pass " << std::endl;
    } else {
        std::cout << __FILE__<< " verfication failed " << std::endl;
    }

    return 0;
}
