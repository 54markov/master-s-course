/*
 * See the Cormen book for details of the following algorithm
 */

#include <algorithm>
#include <iostream>
#include <iterator>
#include <limits>
#include <vector>

template <typename T>
void showMatrix(std::vector<std::vector<T>> &m)
{
    std::cout << std::endl;

    std::for_each(m.begin() + 1, m.end(), [](std::vector<T> &v)
    {
        std::for_each(v.begin() + 1, v.end(), [](T a)
        {
            std::cout << a << "\t";
        });
        std::cout << std::endl;
    });

    std::cout << std::endl;
}

/*
 * Function for printing the optimal
 * parenthesization of a matrix chain product
 */
template <typename T>
void printParenthesis(int                         i,
                      int                         j,
                      int                         n,
                      std::vector<std::vector<T>> bracket,
                      char                        &name)
{
    // If only one matrix left in current segment
    if (i == j)
    {
        std::cout << name++;
        return;
    }
 
    std::cout << "(";
    printParenthesis(i, bracket[i][j], n, bracket, name);
    printParenthesis(bracket[i][j] + 1, j, n, bracket, name);
    std::cout << ")";
}

/*
 * Matrix Ai has dimension p[i-1] x p[i] for i = 1..n
 */
template <typename T>
T matrixChainOrder(const std::vector<T> &p)
{
    /* 
     * m[i,j] = Minimum number of scalar multiplications needed
     * to compute the matrix A[i]A[i + 1]...A[j] = A[i..j] where
     * dimension of A[i] is p[i - 1] x p[i]
     */
    std::vector<std::vector<T>> m;

    /*
     * bracket[i][j] stores optimal break point in
     * subexpression from i to j.
     */
    std::vector<std::vector<T>> bracket;

    // Size of minimum number of scalar multiplications needed matrix
    const int n = p.size();

    for (auto i = 0; i < n; ++i)
    {
        m.push_back(std::vector<T>(n, 0));
        bracket.push_back(std::vector<T>(n, 0));
    }
 
    // 'l' - is chain length
    for (auto l = 2; l < n; ++l)
    {
        for (auto i = 0; i < (n - l + 1); ++i) // Choosing row
        {
            auto j = i + l - 1; // Choosing col
            m[i][j] = std::numeric_limits<T>::max();

            // Finding minimum
            for (auto k = i; k <= (j - 1); ++k)
            {
                // q - cost/scalar multiplications
                auto q = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j];
                if (q < m[i][j])
                {
                    m[i][j] = q;

                    /*
                     * Each entry bracket[i,j]=k shows
                     * where to split the product arr
                     * i,i+1....j for the minimum cost.
                     */
                    bracket[i][j] = k;
                }
            }

            showMatrix(m);
        }
    }
    
    /* 
     * The first matrix is printed as 'A', next as 'B',
     * and so on ...
     */
    auto name = 'A';
    std::cout << "Optimal Parenthesization is : ";
    printParenthesis(1, n - 1, n, bracket, name);
    std::cout << std::endl;

    return m[1][n - 1];
}

void test(const std::vector<int> &v)
{
    auto name = 'A';

    std::cout << "Matrix Chain Multiplications" << std::endl;

    for (auto iter = v.begin() + 1; iter != v.end(); ++iter)
    {
        std::cout << name++ << " [" << *(iter - 1) << ", " << *iter << "]" << std::endl;
    }

    auto result =  matrixChainOrder(v);

    std::cout << "Minimum number of multiplications is: " << result << std::endl;
    std::cout << std::endl;
}

int main(int argc, char const *argv[])
{
    const std::vector<int> v1 = { 1, 10, 5, 2, 20 };
    //const std::vector<int> v2 = { 10, 3, 20, 5, 100, 8 };

    test(v1);

    //test(v2);

    return 0;
}
