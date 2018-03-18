#include <iostream>
#include <vector>
#include <random>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace std;

void generatingMatrix(vector<vector<int>> &a, int row, int col)
{
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, 100);

    cout << "Generating matrix:" << endl;

    for (auto i = 0; i < row; ++i) {
        vector<int> v(col);
        a.push_back(v);
    }
    
    for (auto i = 0; i < row; ++i) {
        for (auto j = 0; j < col; ++j) {
            a[i][j] = distribution(generator);
        }
    }

    for (auto &i : a) {
        for (auto &j : i) {
            cout << j << " ";
        }
        cout << endl;
    }
}

void multiplyMatrix(vector<vector<int>> &a, vector<vector<int>> &b, int m, int n, int q)
{
    vector<vector<int>> c;

    for (auto i = 0; i < m; ++i) {
        vector<int> v(q);
        c.push_back(v);
    }

    cout << "The new matrix is:" << endl;
    
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
 
int main(int argc, char const *argv[])
{
    int m, n; // matrix 1 row, col
    int p, q; // matrix 2 row, col

    vector<vector<int>> matrix1;
    vector<vector<int>> matrix2;

    if (argc < 5) {
        cerr << "usage: m(row1) n(col1) p(row2) q(col2)" << endl;
        return -1;
    }

    m = atoi(argv[1]);
    n = atoi(argv[2]);
    p = atoi(argv[2]);
    q = atoi(argv[4]);

    generatingMatrix(matrix1, m, n);
    generatingMatrix(matrix2, p, q);

    multiplyMatrix(matrix1, matrix2, m, n, q);
 
    return 0;
}