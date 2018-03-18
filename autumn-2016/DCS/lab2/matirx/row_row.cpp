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

using namespace std;

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

    //cout << "The new matrix is:" << endl;
    
    for (auto i = 0; i < m; ++i) {
        for (auto k = 0; k < q; ++k) {
            for (auto j = 0; j < n; ++j) {
                c[i][j] = c[i][j] + (a[i][k]*b[k][j]);
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
 
int main(int argc, char const *argv[])
{
    std::string line;
    std::ifstream myfile("workfile");
    
    if (!myfile.is_open()) {
        cerr << "can't open file" << endl;
        return -1;
    }

    while (getline(myfile,line)) {
        std::vector<vector<int>> matrix1;
        std::vector<vector<int>> matrix2;
        
        int param = stoi(line);
        
        generatingMatrix(matrix1, param, param);
        generatingMatrix(matrix2, param, param);

        chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();

        multiplyMatrix(matrix1, matrix2, param, param, param);

        chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();

        chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double>>(t2 - t1);

        std::cout << __FILE__<< " It took on "<< param << " - " << time_span.count() << " seconds." << std::endl;

    }
    myfile.close();
 
    return 0;
}